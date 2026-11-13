"""Hybrid retrieval: BM25 (always) + bge-m3 dense vectors + optional Qwen3 reranker.

BM25 keeps working without any API and is responsible for the "nothing relevant"
decision on out-of-scope questions. Dense search adds paraphrases that share few
words with the knowledge base. Results are fused with Reciprocal Rank Fusion.
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from context_builder import ContextBuilder
from llm_client import ApiError, env_number
from search_kb import SearchIndex, prepare_query


@dataclass
class Retrieved:
    fragments: list
    debug: dict = field(default_factory=dict)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class HybridRetriever:
    def __init__(self, fragments, client=None, cache_dir=None):
        self.index = SearchIndex(fragments)
        self.fragments = self.index.fragments
        self.client = client
        self.vectors = None
        self.top_k = env_number("TOP_K", 4, 1, 12)
        self.dense_min = env_number("DENSE_MIN", 0.50, 0.0, 1.0, float)
        self.dense_alone_min = env_number("DENSE_ALONE_MIN", 0.62, 0.0, 1.0, float)
        self.use_reranker = os.environ.get("USE_RERANKER", "1").strip() not in {"0", "false", "no"}
        self.rerank_min = env_number("RERANK_MIN", 0.05, 0.0, 1.0, float)
        self.context_mode = os.environ.get("CONTEXT_MODE", "sections").strip().lower()
        if self.context_mode not in {"sections", "fragments"}:
            raise ValueError("CONTEXT_MODE must be 'sections' or 'fragments'.")
        self.context_builder = ContextBuilder(
            self.index, env_number("MAX_CONTEXT_CHARS", 16000, 4000, 64000))
        if client is not None and client.settings.has_embeddings:
            try:
                self.vectors = self._load_vectors(cache_dir)
            except ApiError as error:
                print(f"Эмбеддинги недоступны ({error}); работает только BM25.", file=sys.stderr)

    @staticmethod
    def _doc_text(fragment):
        return f"{fragment.heading}\n{fragment.text}"

    def _load_vectors(self, cache_dir):
        texts = [self._doc_text(f) for f in self.fragments]
        digest = hashlib.sha256(
            (self.client.settings.emb_model + "\0" + "\0".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        path = Path(cache_dir) / f"embeddings-{digest}.json" if cache_dir else None
        if path and path.is_file():
            try:
                vectors = json.loads(path.read_text(encoding="utf-8"))
                if len(vectors) == len(texts):
                    return vectors
            except ValueError:
                pass
        vectors = self.client.embed(texts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vectors), encoding="utf-8")
        return vectors

    @property
    def mode(self):
        parts = ["bm25"]
        if self.vectors is not None:
            parts.append("bge-m3")
        if self.use_reranker and self.client is not None and self.client.settings.has_reranker:
            parts.append("reranker")
        return "+".join(parts)

    def search(self, question, limit=None):
        """Retrieve initial hits, then assemble evidence within a character budget.

        `limit` controls initial hits; sections mode may return more sections.
        question is plain text or a 'Пользователь: ... Ассистент: ...' transcript.
        """
        limit = limit or self.top_k
        timings = {}
        started = time.perf_counter()
        bm25 = self.index.rank(question, limit=20)
        timings["bm25_ms"] = round((time.perf_counter() - started) * 1000, 1)
        bm25_scores = {hit.index: hit.score for hit in bm25}

        dense_scores = {}
        if self.vectors is not None:
            started = time.perf_counter()
            try:
                query = prepare_query(question).semantic_text
                qvec = self.client.embed([query])[0]
                scored = sorted(((_dot(qvec, v), i) for i, v in enumerate(self.vectors)), reverse=True)[:20]
                dense_scores = {i: s for s, i in scored if s >= self.dense_min}
                if not bm25 and (not scored or scored[0][0] < self.dense_alone_min):
                    dense_scores = {}  # out of scope: neither lexical nor strong semantic evidence
            except ApiError as error:
                timings["dense_error"] = str(error)
            timings["dense_ms"] = round((time.perf_counter() - started) * 1000, 1)

        fused = {}
        for ranking in (sorted(bm25_scores, key=bm25_scores.get, reverse=True),
                        sorted(dense_scores, key=dense_scores.get, reverse=True)):
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        order = sorted(fused, key=lambda i: (-fused[i], i))

        rerank_scores = {}
        if (order and self.use_reranker and self.client is not None
                and self.client.settings.has_reranker and len(order) > 1):
            started = time.perf_counter()
            candidates = order[:12]
            try:
                parsed = prepare_query(question)
                context = " ".join(text for role, text in parsed.previous
                                   if role in {"пользователь", "клиент", "user"})[-400:]
                query = f"{parsed.latest} (контекст: {context})" if context else parsed.latest
                ranked = self.client.rerank(query, [self._doc_text(self.fragments[i]) for i in candidates],
                                            top_n=len(candidates))
                rerank_scores = {candidates[i]: s for i, s in ranked if 0 <= i < len(candidates)}
                ranked_ids = sorted(rerank_scores, key=rerank_scores.get, reverse=True)
                # Score scales differ between reranker deployments: always keep the two best.
                kept = [i for n, i in enumerate(ranked_ids) if n < 2 or rerank_scores[i] >= self.rerank_min]
                order = kept or order
            except ApiError as error:
                timings["rerank_error"] = str(error)
            timings["rerank_ms"] = round((time.perf_counter() - started) * 1000, 1)

        fragments = self.index.expand(order, limit)
        context_debug = {"mode": self.context_mode}
        if self.context_mode == "sections":
            started = time.perf_counter()
            fragments, context_stats = self.context_builder.build(question, fragments)
            context_debug.update(context_stats)
            timings["context_ms"] = round((time.perf_counter() - started) * 1000, 1)
        candidates = []
        for i in order[:limit * 2]:
            fragment = self.fragments[i]
            candidates.append({
                "source": fragment.source, "section": fragment.heading,
                "bm25": round(bm25_scores.get(i, 0.0), 2),
                "dense": round(dense_scores.get(i, 0.0), 3),
                "rerank": round(rerank_scores[i], 3) if i in rerank_scores else None,
            })
        return Retrieved(fragments, {"mode": self.mode, "timings": timings,
                                    "candidates": candidates, "context": context_debug})
