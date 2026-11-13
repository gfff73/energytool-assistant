"""Minimal OpenAI-compatible client (chat, embeddings, rerank) on the standard library.

The platform provides one API for Qwen (generation), BAAI/bge-m3 (embeddings) and
Qwen3 Reranker. Only the standard library is used so that the hackathon ZIP has no
dependencies; the web service reuses the same client.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


THINK = re.compile(r"<think>.*?</think>\s*", re.S)
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def env(*names, default=""):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value and value != name:
            return value
    return default


def env_number(name, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name, default=str(default)))))
    except ValueError:
        return default


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        # Never forward the Authorization header to another host.
        return None


class ApiError(RuntimeError):
    pass


class IncompleteAnswer(ApiError):
    """Generation stopped before a complete final answer was available."""


@dataclass(frozen=True, repr=False)
class Settings:
    base_url: str = ""
    llm_key: str = ""
    emb_key: str = ""
    rerank_key: str = ""
    llm_model: str = ""
    emb_model: str = ""
    rerank_model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 60.0
    extra_body: tuple = ()

    @classmethod
    def from_environment(cls):
        base_url = env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
        if base_url:
            address = urlsplit(base_url)
            if (address.scheme not in {"http", "https"} or not address.hostname
                    or address.username or address.password or address.query or address.fragment):
                raise ValueError("LLM_BASE_URL должен быть адресом HTTP(S) API без ключей и параметров.")
        llm_key = env("LLM_API_KEY", "OPENAI_API_KEY")
        extra = env("LLM_EXTRA_BODY")
        try:
            extra_body = tuple(json.loads(extra).items()) if extra else ()
        except (ValueError, AttributeError):
            raise ValueError("LLM_EXTRA_BODY должен быть JSON-объектом.") from None
        return cls(
            base_url=base_url,
            llm_key=llm_key,
            emb_key=env("EMBEDDER_API_KEY", default=llm_key),
            rerank_key=env("RERANKER_API_KEY", default=llm_key),
            llm_model=env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL"),
            emb_model=env("EMB_MODEL"),
            rerank_model=env("RERANKER_MODEL"),
            max_tokens=env_number("MAX_TOKENS", 1024, 64, 8192),
            temperature=env_number("TEMPERATURE", 0.0, 0.0, 2.0, float),
            timeout=env_number("LLM_TIMEOUT", 60.0, 5.0, 300.0, float),
            extra_body=extra_body,
        )

    @property
    def has_llm(self):
        return bool(self.base_url and self.llm_key and self.llm_model)

    @property
    def has_embeddings(self):
        return bool(self.base_url and self.emb_key and self.emb_model)

    @property
    def has_reranker(self):
        return bool(self.base_url and self.rerank_key and self.rerank_model)


class Client:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_environment()
        self.opener = build_opener(NoRedirect())

    def _post(self, endpoint, key, body, attempts=3, timeout=None):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.settings.base_url + endpoint, data=data, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        for attempt in range(attempts):
            try:
                with self.opener.open(request, timeout=timeout or self.settings.timeout) as response:
                    return json.loads(response.read())
            except HTTPError as error:
                status = error.code
                error.close()
                if status in RETRYABLE and attempt + 1 < attempts:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise ApiError(f"{endpoint}: HTTP {status}") from None
            except (URLError, TimeoutError, OSError, ValueError) as error:
                if attempt + 1 < attempts:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise ApiError(f"{endpoint}: {type(error).__name__}") from None

    def chat(self, messages, max_tokens=None, temperature=None):
        s = self.settings
        if not s.has_llm:
            raise ApiError("LLM не настроена")
        body = {
            "model": s.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or s.max_tokens,
            "temperature": s.temperature if temperature is None else temperature,
        }
        body.update(dict(s.extra_body))
        result = self._post("/chat/completions", s.llm_key, body)
        try:
            choice = result["choices"][0]
            answer = choice["message"].get("content") or ""
            reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ApiError("chat: неожиданный формат ответа") from None
        if reason == "length":
            raise IncompleteAnswer("chat: ответ оборван по лимиту токенов")
        if reason not in {None, "stop"}:
            raise ApiError("chat: генерация завершена без обычного текстового ответа")
        if not isinstance(answer, str):
            raise ApiError("chat: content должен быть строкой")
        answer = THINK.sub("", answer)
        if "<think>" in answer:  # unterminated reasoning block: keep what follows it, if anything
            answer = answer.split("</think>")[-1] if "</think>" in answer else ""
        answer = answer.strip()
        if not answer:
            raise IncompleteAnswer("chat: пустой ответ модели")
        return answer

    def embed(self, texts, batch=32):
        s = self.settings
        if not s.has_embeddings:
            raise ApiError("Эмбеддер не настроен")
        vectors = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            result = self._post("/embeddings", s.emb_key, {"model": s.emb_model, "input": chunk})
            part = [None] * len(chunk)
            for i, item in enumerate(result["data"]):
                vector = [float(x) for x in item["embedding"]]
                norm = sum(x * x for x in vector) ** 0.5 or 1.0
                part[int(item.get("index", i))] = [x / norm for x in vector]
            if any(v is None for v in part):
                raise ApiError("embeddings: получены не все векторы")
            vectors.extend(part)
        return vectors

    def rerank(self, query, documents, top_n):
        s = self.settings
        if not s.has_reranker:
            raise ApiError("Реранкер не настроен")
        result = self._post("/rerank", s.rerank_key, {
            "model": s.rerank_model, "query": query, "documents": documents, "top_n": top_n,
        }, attempts=2, timeout=min(s.timeout, 30))
        entries = result.get("results", result.get("data", []))
        return [(int(e["index"]), float(e.get("relevance_score", e.get("score", 0.0)))) for e in entries]
