"""Local retrieval checks. Demo questions stay in ignored materials, never in the ZIP.

Targets are manually selected sections/anchors, not LLM-labelled answers. Multiple
targets in one group are alternatives. Recall@k averages the fraction of required
groups retrieved per question. This small development set is not a hidden test.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import search_kb

ROOT = Path(__file__).resolve().parent


def target(source, heading="", text=""):
    return {"source": source, "heading": heading, "text": text}


DEMO_TARGETS = [
    [target("23-solar.md", "Размещение", "запад-восток")],
    [target("23-solar.md", "Размещение", "вертикальные")],
    [target("23-solar.md", "Целевой диапазон мощности", "наращивается"), target("26-general.md", text="мощность наращивается")],
    [target("23-solar.md", text="шифера, которому"), target("23-solar.md", text="шифер старше"), target("02-product-scope-and-limits.md", text="шифер")],
    [target("22-askue.md", text="оператор")],
    [target("22-askue.md", "Кому подходит", "цехам")],
    [target("22-askue.md", "Кому подходит", "ретейл")],
    [target("25-meter-reading.md", "Что распознаётся", "тарифам")],
    [target("25-meter-reading.md", "Что распознаётся", "Год производства")],
    [target("14-tariffs.md", "Что именно растёт", "потреблен")],
    [target("17-faq.md", "реактивная энергия")],
    [target("21-orem.md", text="аудит существующей системы")],
    [target("21-orem.md", "невыгодно"), target("26-general.md", "невыгодно")],
    [target("23-solar.md", "Лучший профиль клиента"), target("23-solar.md", text="преимущественно ночью")],
    [target("22-askue.md", "Кому подходит", "арендаторам"), target("03-calculation-and-requests.md", "Для платформы учёта")],
    [target("24-losses.md", text="суточн"), target("03-calculation-and-requests.md", "Для поиска коммерческих потерь")],
    [target("14-tariffs.md", "Что именно растёт"), target("11-price-categories.md", "Шесть категорий")],
    [target("03-calculation-and-requests.md", "Для платформы учёта")],
]

EXTRA_CASES = [
    ("Как долго хранятся фотографии счётчиков?", [target("25-meter-reading.md", "Сколько хранятся фотографии")]),
    ("Есть демонстрационный доступ к платформе учёта?", [target("22-askue.md", "Как посмотреть до покупки"), target("22-askue.md", "Можно ли сначала попробовать")]),
    ("Для оптового рынка нужно 750 кВт или 5 МВт?", [target("12-power-thresholds.md", "Пороги оптового рынка"), target("21-orem.md", "Порог доступа")]),
    ("Учитывает ли платформа воду?", [target("22-askue.md", "Работает ли учёт воды")]),
    ("Какая точность распознавания показаний электросчётчиков по фотографии?", [target("25-meter-reading.md", "Точность"), target("25-meter-reading.md", "Насколько точно")]),
    ("Пользователь: Хотим солнечные панели.\nАссистент: На какой крыше?\nПользователь: Теперь другой вопрос: сколько хранятся фотографии счётчиков?", [target("25-meter-reading.md", "Сколько хранятся фотографии")]),
    ("Сколько калорий в солнечном манго?", []),
    ("Как настроить пароль Wi-Fi на роутере?", []),
    ("Какая погода завтра в Москве?", []),
    ("Как приготовить электричество из макарон?", []),
    ("Хочу отслеживать машины по GPS", []),
    ("Напиши стих про счётчик и дракона", []),
]


def matches(fragment, expected):
    normalize = lambda text: " ".join(text.casefold().replace("ё", "е").split())
    return (fragment.source.endswith(expected["source"])
            and normalize(expected["heading"]) in normalize(fragment.heading)
            and normalize(expected["text"]) in normalize(fragment.text))


def load_cases():
    with (ROOT / "materials/Данные/demo_questions.csv").open(encoding="utf-8-sig", newline="") as stream:
        demos = list(csv.DictReader(stream))
    if len(demos) != len(DEMO_TARGETS):
        raise ValueError("Разметка рассчитана на 18 демонстрационных вопросов.")
    cases = [("demo-" + row["id"], row["question"], [targets]) for row, targets in zip(demos, DEMO_TARGETS, strict=True)]
    cases.extend((f"extra-{i}", question, [targets] if targets else [])
                 for i, (question, targets) in enumerate(EXTRA_CASES, 1))
    return cases


def evaluate(module, k=3):
    fragments = module.load_fragments(search_kb.KNOWLEDGE_BASE)
    cases = load_cases()
    for case_id, _, groups in cases:
        for group in groups:
            if not any(matches(fragment, label) for fragment in fragments for label in group):
                raise ValueError(f"Нет эталонного фрагмента: {case_id}")
    index = module.SearchIndex(fragments) if hasattr(module, "SearchIndex") else fragments
    rows = []
    for case_id, question, groups in cases:
        found = index.find(question, limit=k) if hasattr(index, "find") else module.find_fragments(question, index, limit=k)
        hits = sum(any(matches(fragment, label) for fragment in found for label in group) for group in groups)
        rows.append({"id": case_id, "answerable": bool(groups),
                     "recall": hits / len(groups) if groups else None,
                     "abstained": not found,
                     "sources": [{"source": f.source, "heading": f.heading} for f in found]})
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    return {"cases": len(rows), "positive_cases": len(positives), "negative_cases": len(negatives),
            f"recall@{k}": sum(r["recall"] for r in positives) / len(positives),
            "negative_abstention": sum(r["abstained"] for r in negatives) / len(negatives),
            "positive_abstention": sum(r["abstained"] for r in positives) / len(positives), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Проверка поиска: 24 вопроса с источниками и 6 без ответа")
    parser.add_argument("--baseline", type=Path, help="Путь к сохранённой предыдущей версии search_kb.py")
    parser.add_argument("--output", type=Path, help="JSON-отчёт, например tmp/search-evaluation.json")
    args = parser.parse_args()
    module = search_kb
    if args.baseline:
        spec = importlib.util.spec_from_file_location("search_baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    result = evaluate(module)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Вопросов: {result['cases']}; recall@3: {result['recall@3']:.1%}; "
          f"правильный отказ: {result['negative_abstention']:.1%}; "
          f"пустой поиск на вопросах с ответом: {result['positive_abstention']:.1%}")
    for row in result["rows"]:
        if row["recall"] == 0 or (not row["answerable"] and not row["abstained"]):
            print("MISS", row["id"], "; ".join(f"{f['source']} / {f['heading']}" for f in row["sources"]))


if __name__ == "__main__":
    main()
