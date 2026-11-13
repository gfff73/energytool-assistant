"""Hackathon entry point: questions.csv + knowledge-base.zip -> submission.csv."""

import argparse
import csv
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from consultant import Consultant, parse_dialog
from llm_client import Client, Settings
from retrieval import HybridRetriever
from search_kb import parse_document


SERVICE_ERROR = (
    "Сейчас не удалось получить ответ от сервиса консультации. "
    "Пожалуйста, повторите вопрос позже или свяжитесь со специалистом Energytool."
)


def find_input(directory, *names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise ValueError(f"В {directory} не найден входной файл {names[0]}")


def read_questions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "question"}.issubset(reader.fieldnames or []):
            raise ValueError("Во входном CSV нужны столбцы id и question.")
        questions = []
        seen = set()
        for row in reader:
            question_id = row["id"]
            if not question_id or not question_id.strip() or question_id in seen:
                raise ValueError("Во входном CSV есть пустые или повторяющиеся ID.")
            if row["question"] is None or None in row:
                raise ValueError("Некорректное число полей во входном CSV.")
            seen.add(question_id)
            questions.append((question_id, row["question"]))
    if not questions:
        raise ValueError("Входной CSV не содержит вопросов.")
    return questions


def read_knowledge_base(path):
    fragments = []
    with zipfile.ZipFile(path) as archive:
        for entry in sorted(archive.infolist(), key=lambda entry: entry.filename):
            source = Path(entry.filename)
            if entry.is_dir() or "__MACOSX" in source.parts:
                continue
            if source.suffix.lower() in {".md", ".txt"}:
                text = archive.read(entry).decode("utf-8-sig")
                fragments.extend(parse_document(entry.filename, text))
    if not fragments:
        raise ValueError("В архиве базы знаний нет непустых документов Markdown или TXT.")
    return fragments


def positive_setting(name, default, maximum):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} должен быть целым числом.") from None
    if value < 1:
        raise ValueError(f"{name} должен быть положительным.")
    return min(value, maximum)


def write_submission(questions, answers):
    output = Path.cwd() / "submission.csv"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent,
            prefix=".submission-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (question_id, _), answer in zip(questions, answers, strict=True):
                if not answer or not answer.strip() or len(answer) > 10000:
                    raise ValueError("Ответ пустой или превышает 10 000 символов.")
                writer.writerow([question_id, answer])
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def run(directory):
    questions_path = find_input(directory, "questions.csv", "train.csv", "demo_questions.csv")
    kb_path = find_input(directory, "knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    questions = read_questions(questions_path)
    fragments = read_knowledge_base(kb_path)
    settings = Settings.from_environment()
    if not settings.has_llm:
        raise ValueError("Задайте LLM_BASE_URL, LLM_API_KEY и LLM_MODEL.")
    client = Client(settings)
    retriever = HybridRetriever(fragments, client)
    consultant = Consultant(retriever, client)
    workers = positive_setting("MAX_WORKERS", 3, 8)
    print(f"Поиск: {retriever.mode}.", flush=True)

    def answer_question(question):
        history, message = parse_dialog(question)
        if not message.strip():
            return "Опишите, пожалуйста, какую задачу вы хотите решить с помощью Energytool.", "empty", False
        result = consultant.answer(history, message)
        if result.debug.get("generation") != "llm":
            # Keep an extractive answer instead of an error text, but report the failure.
            print(f"LLM недоступна: {result.debug.get('llm_error')}", file=sys.stderr)
        return result.text, result.debug.get("generation", "unknown"), bool(result.debug.get("llm_retry"))

    print(f"Вопросов: {len(questions)}. Фрагментов базы: {len(fragments)}.", flush=True)
    answers = [None] * len(questions)
    failures = 0
    generations = {}
    retried = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(answer_question, question): index
            for index, (_, question) in enumerate(questions)
        }
        for future in as_completed(pending):
            index = pending[future]
            try:
                answers[index], generation, was_retried = future.result()
                generations[generation] = generations.get(generation, 0) + 1
                retried += int(was_retried)
            except (RuntimeError, OSError, ValueError) as error:
                failures += 1
                answers[index] = SERVICE_ERROR
                print(f"Ошибка сервиса для строки {index + 1}: {error}", file=sys.stderr)
    output = write_submission(questions, answers)
    print(f"Создан {output.name}. Ответов: {len(answers)}. Ошибок сервиса: {failures}.")
    print(f"Ответов LLM: {generations.get('llm', 0)}. Запасных ответов: {generations.get('fallback', 0)}. "
          f"Повторов неполной генерации: {retried}.")


def main():
    parser = argparse.ArgumentParser(description="Создать submission.csv для проверки на платформе")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR") or "data"))
    args = parser.parse_args()
    try:
        run(args.data_dir)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Ошибка запуска: {error}\n")


if __name__ == "__main__":
    main()
