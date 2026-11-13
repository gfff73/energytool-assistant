import argparse
import re
from collections import Counter
from dataclasses import dataclass
from math import isfinite, log1p
from pathlib import Path
from textwrap import wrap

from russian_stemmer import stem
from thesaurus import expansions


KNOWLEDGE_BASE = Path(__file__).resolve().parent / "materials/knowledge-base"
FRAGMENT_LENGTH = 900
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)
WORD = re.compile(r"[^\W_]+")
SIM_CARD = re.compile(r"\b(?:сим|sim)[ -]?(?:карт[а-я]*|к[аеиу])\b")
TURN = re.compile(r"^\s*(пользователь|клиент|user|ассистент|помощник|assistant|бот)\s*:\s*", re.I | re.M)
STOP_WORDS = set(
    "без был была были будет вам вас ваш ваша вашей ваши где для или как какие какой "
    "когда кто мне можно надо нам нас наш наши него нет них она они оно под при про "
    "сами себе свой свои так такие такой там тоже чем через что чтобы это этот эту "
    "есть если еще уже только самый нужно хотим хочу можем сможет стоит вообще "
    "какая какое каких сколько почему зачем чего какой-то кто-то ли оба ваша вашем "
    "ваших нашей наших сейчас пока примерно таких всего сначала затем пожалуйста "
    "можете объяснить расскажите теперь другой вопрос то этом этих такой-то "
    "потом начать кажется".split()
)


@dataclass(frozen=True)
class Fragment:
    source: str
    heading: str
    text: str


def split_text(text):
    """Keep nearby paragraphs together, with a limit on fragment length."""
    pending = ""
    for paragraph in re.split(r"\n[ \t]*\n", text.strip()):
        if not paragraph.strip():
            continue
        if len(paragraph) > FRAGMENT_LENGTH:
            if pending:
                yield pending
                pending = ""
            yield from wrap(paragraph, FRAGMENT_LENGTH, replace_whitespace=False,
                            break_long_words=False, break_on_hyphens=False)
        elif pending and len(pending) + 2 + len(paragraph) > FRAGMENT_LENGTH:
            yield pending
            pending = paragraph
        else:
            pending = f"{pending}\n\n{paragraph}" if pending else paragraph
    if pending:
        yield pending


def parse_document(source, text):
    fragments = []
    sections = HEADING.split(text)
    fragments.extend(Fragment(source, Path(source).stem, part) for part in split_text(sections[0]))
    headings = []
    for level, title, body in zip(sections[1::3], sections[2::3], sections[3::3]):
        while headings and headings[-1][0] >= len(level):
            headings.pop()
        headings.append((len(level), title.strip()))
        heading = " / ".join(title for _, title in headings)
        fragments.extend(Fragment(source, heading, part) for part in split_text(body))
    return fragments


def load_fragments(directory):
    fragments = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            source = path.relative_to(directory).as_posix()
            fragments.extend(parse_document(source, path.read_text(encoding="utf-8-sig")))
    return fragments


def tokens(text):
    normalized = SIM_CARD.sub("симкарта", text.casefold().replace("ё", "е"))
    normalized = re.sub(r"\bфото\b", "фотография", normalized)
    return [
        stem(word)
        for word in WORD.findall(normalized)
        if (len(word) >= 3 or word.isdecimal()) and word not in STOP_WORDS
    ]


def term_aliases(text):
    """stem -> {stem, strong thesaurus synonyms of that word}."""
    normalized = SIM_CARD.sub("симкарта", text.casefold().replace("ё", "е"))
    aliases = {}
    for word in WORD.findall(normalized):
        if (len(word) < 3 and not word.isdecimal()) or word in STOP_WORDS:
            continue
        term = stem(word)
        aliases.setdefault(term, {term}).update(tokens(expansions(word, strong_only=True)))
    return aliases


@dataclass(frozen=True)
class SearchQuery:
    latest: str
    previous: tuple

    @property
    def semantic_text(self):
        context = "\n".join(text for _, text in self.previous)
        return f"Контекст: {context}\nПоследний вопрос: {self.latest}" if context else self.latest


DATA_REQUEST = re.compile(
    r"\b(?:что\s+(?:вам\s+)?(?:нужн\w*|понадоб\w*|потреб\w*|необходим\w*)"
    r"|каки\w*\s+(?:сведени\w*|данн\w*|документ\w*|параметр\w*|информаци\w*)"
    r"\s+(?:(?:вам|надо|нужно)\s+)?"
    r"(?:нужн\w*|понадоб\w*|потреб\w*|необходим\w*|подготов\w*|предостав\w*|собрать|прислать))",
    re.I,
)
ESTIMATE_PURPOSE = re.compile(
    r"\bдля\s+(?:(?:предварительн\w*|так\w*|ваш\w*|эт\w*)\s+)?"
    r"(?:оценки|расч[её]та)(?:\s+(?:стоимости|проекта|подключения|объекта))?\s*[?.!]*\s*$",
    re.I,
)
OTHER_INTENT = re.compile(r"\b(?:сколько|когда|почему|зачем|где|поч[её]м|срок\w*|хран\w*|персональн\w*)\b", re.I)
REQUEST_PRODUCTS = (
    ("солнечной станции", re.compile(r"солнечн\w*\s+(?:станц|панел|электростанц)|\bсэс\b", re.I)),
    ("выхода на оптовый рынок", re.compile(r"оптов\w*\s+рын|\bорэм\b", re.I)),
    ("платформы учёта", re.compile(
        r"\bаскуэ\b|(?:автоматическ\w*|автоматизированн\w*|дистанционн\w*)\s+уч[её]т"
        r"|платформ\w*\s+(?:для\s+)?уч[её]т|автоматиз\w*\s+сбор\s+показан", re.I)),
    ("поиска коммерческих потерь", re.compile(
        r"коммерческ\w*\s+потер|(?:поиск|найти|находить|выявлять)\s+(?:коммерческ\w*\s+)?потер", re.I)),
    ("распознавания показаний", re.compile(
        r"распознаван\w*\s+(?:показан|фотограф)|(?:показан\w*|сч[её]тчик\w*)\s+по\s+фото", re.I)),
)


def focus_data_request(latest, previous):
    """Resolve one narrow follow-up without changing the client's actual message.

    Only requests for the inputs to an estimate are rewritten. An explicit,
    unambiguous product must occur in the user's own messages. Multiple products
    or additional questions keep the original query and its context.
    """
    if (not previous or not DATA_REQUEST.search(latest) or not ESTIMATE_PURPOSE.search(latest)
            or latest.count("?") > 1 or OTHER_INTENT.search(latest)):
        return None
    user_context = [text for role, text in previous if role in {"пользователь", "клиент", "user"}]
    if not user_context:
        return None
    context = "\n".join([*user_context, latest])
    products = [name for name, pattern in REQUEST_PRODUCTS if pattern.search(context)]
    if len(products) != 1:
        return None
    return f"Какие сведения могут понадобиться для {products[0]}?"


def prepare_query(question):
    """Keep the latest user turn dominant; retain a bounded context for ellipses."""
    markers = list(TURN.finditer(question))
    turns = [(m.group(1).casefold(), question[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(question)].strip())
             for i, m in enumerate(markers)]
    users = [i for i, (role, _) in enumerate(turns) if role in {"пользователь", "клиент", "user"}]
    if not users:
        return SearchQuery(question.strip(), ())
    last = users[-1]
    latest = turns[last][1]
    if re.search(r"другой вопрос|сменим тему|теперь (?:про|о\b)", latest, re.I):
        return SearchQuery(latest, ())
    previous = tuple((role, text[-700:]) for role, text in turns[max(0, last - 4):last])
    focused = focus_data_request(latest, previous)
    if focused:
        return SearchQuery(focused, ())
    # A short value plus a suitability question contains no subject of its own.
    # Restore it from user statements, without turning an assistant's guess
    # into a customer fact. Explicit new topics fail this narrow vocabulary gate.
    words = WORD.findall(latest.casefold().replace("ё", "е"))
    value_word = re.compile(r"\d+(?:квт|мвт)?|полгода|полтора|полторы|один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять")
    other_word = re.compile(r"год\w*|лет|месяц\w*|квт|мвт|всего|около|только|примерно|да|нет|этого|это|так|будет|достаточно|хватит|подойдет|можно|уже|нам|у|нас")
    if (previous and 0 < len(words) <= 14 and any(value_word.fullmatch(w) for w in words)
            and any(w in {"хватит", "достаточно", "подойдет"} for w in words)
            and all(value_word.fullmatch(w) or other_word.fullmatch(w) for w in words)):
        user_statements = [text for role, text in previous if role in {"пользователь", "клиент", "user"}]
        if user_statements:
            return SearchQuery("\n".join([*user_statements, latest]), previous)
    return SearchQuery(latest, previous)


@dataclass(frozen=True)
class SearchHit:
    index: int
    fragment: Fragment
    score: float
    coverage: float


class SearchIndex:
    """BM25 with positive IDF, built once and reused across concurrent questions.

    Thresholds are conservative development defaults, not probabilities. They
    need reevaluation on an independent labelled set when the corpus changes.
    """

    def __init__(self, fragments, min_score=3.0, min_coverage=0.10, relative_cutoff=0.4, expansion_weight=0.7,
                 k1=1.5, b=0.75, heading_boost=0.6, leaf_boost=1.0):
        if (not all(isfinite(v) for v in (min_score, min_coverage, relative_cutoff))
                or min_score < 0 or not 0 <= min_coverage <= 1 or not 0 <= relative_cutoff <= 1):
            raise ValueError("Некорректные пороги поиска.")
        self.fragments = tuple(fragments)
        self.min_score, self.min_coverage, self.relative_cutoff = min_score, min_coverage, relative_cutoff
        self.expansion_weight = expansion_weight
        self.k1, self.b, self.heading_boost, self.leaf_boost = k1, b, heading_boost, leaf_boost
        self.counts = tuple(Counter(tokens(f.heading + "\n" + f.text)) for f in self.fragments)
        self.body_counts = tuple(Counter(tokens(f.text)) for f in self.fragments)
        self.heading_terms = tuple(set(tokens(f.heading)) for f in self.fragments)
        self.leaf_terms = tuple(set(tokens(f.heading.split(" / ")[-1])) for f in self.fragments)
        self.lengths = tuple(sum(c.values()) for c in self.body_counts)
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1
        self.average_length = self.average_length or 1
        occurrences = Counter(term for counter in self.counts for term in counter)
        size = len(self.fragments)
        self.idf = {term: log1p((size - count + 0.5) / (count + 0.5)) for term, count in occurrences.items()}
        self.unknown_idf = log1p((size + 0.5) / 0.5)

    def query_weights(self, query):
        """Original terms (weight 1 for the latest turn, decaying for context) + thesaurus terms."""
        weights = {term: 1.0 for term in tokens(query.latest)}
        for position, (role, text) in enumerate(reversed(query.previous)):
            weight = (0.65 if role in {"пользователь", "клиент", "user"} else 0.18) * (0.9 ** position)
            for term in tokens(text):
                weights[term] = max(weights.get(term, 0), weight)
        original = set(weights)
        if self.expansion_weight:
            for term in tokens(expansions(query.latest)):
                weights[term] = max(weights.get(term, 0), self.expansion_weight)
            for role, text in query.previous:
                if role in {"пользователь", "клиент", "user"}:
                    for term in tokens(expansions(text)):
                        weights[term] = max(weights.get(term, 0), self.expansion_weight * 0.5)
        weights = {term: weight * (0.35 if term.isdecimal() else 1) for term, weight in weights.items()}
        return weights, original

    def rank(self, question, limit=30, filter_relevance=True):
        if limit <= 0 or not self.fragments:
            return []
        query = prepare_query(question)
        weights, original = self.query_weights(query)
        if not weights:
            return []
        denominator = sum(self.idf.get(term, self.unknown_idf) * weights[term] for term in original) or 1.0
        # An original query term counts as "covered" when it, or one of its strong thesaurus
        # synonyms, occurs in the fragment. Weak synonyms only influence ranking.
        aliases = term_aliases(query.latest)
        for _, text in query.previous:
            for term, alias in term_aliases(text).items():
                aliases.setdefault(term, alias)
        primary = {term for term in tokens(query.latest) if not term.isdecimal()}
        useful_original = [t for t in original if not t.isdecimal()]
        hits = []
        for index, counter in enumerate(self.counts):
            matched = counter.keys() & weights.keys()
            if not matched:
                continue
            # BM25 term-frequency saturation and document-length normalization.
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
            body = self.body_counts[index]
            score = sum(self.idf[term] * weights[term] * (
                body[term] * (self.k1 + 1) / (body[term] + norm)
                + self.heading_boost * (term in self.heading_terms[index])
                + self.leaf_boost * (term in self.leaf_terms[index])
            ) for term in sorted(matched))
            covered = {t for t in original if aliases.get(t, {t}) & counter.keys()}
            coverage = sum(self.idf.get(t, self.unknown_idf) * weights[t] for t in covered) / denominator
            enough_terms = len([t for t in covered if not t.isdecimal()]) >= min(2, len(useful_original))
            latest_match = not primary or bool(primary & covered)
            if filter_relevance and (score < self.min_score or coverage < self.min_coverage or not enough_terms or not latest_match):
                continue
            hits.append(SearchHit(index, self.fragments[index], score, coverage))
        hits.sort(key=lambda hit: (-hit.score, hit.index))
        if filter_relevance and hits:
            cutoff = hits[0].score * self.relative_cutoff
            hits = [hit for hit in hits if hit.score >= cutoff]
        return hits[:limit]

    def find(self, question, limit=3):
        return self.expand([hit.index for hit in self.rank(question, len(self.fragments))], limit)

    def expand(self, indices, limit=3):
        """Return distinct sections; include adjoining chunks to preserve tables."""
        result, used = [], set()
        for index in indices:
            if len(result) >= limit:
                break
            fragment = self.fragments[index]
            key = (fragment.source, fragment.heading)
            if key in used:
                continue
            used.add(key)
            start = end = index
            length = len(fragment.text)
            # At most two adjacent chunks, never text from another heading.
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(self.fragments):
                    candidate = self.fragments[neighbour]
                    if ((candidate.source, candidate.heading) == key
                            and length + len(candidate.text) + 2 <= 2800):
                        start, end = min(start, neighbour), max(end, neighbour)
                        length += len(candidate.text) + 2
            result.append(Fragment(fragment.source, fragment.heading,
                                   "\n\n".join(f.text for f in self.fragments[start:end + 1])))
        return result


def find_fragments(question, fragments, limit=3):
    index = fragments if isinstance(fragments, SearchIndex) else SearchIndex(fragments)
    return index.find(question, limit)


def main():
    parser = argparse.ArgumentParser(description="Поиск фрагментов в базе знаний Energytool")
    parser.add_argument("question", nargs="?", help="Вопрос, по которому нужно найти документы")
    args = parser.parse_args()
    fragments = load_fragments(KNOWLEDGE_BASE)
    if not fragments:
        parser.exit(1, f"Не найдены документы в папке {KNOWLEDGE_BASE}\n")

    question = args.question if args.question is not None else input("Вопрос: ")
    found = find_fragments(question, fragments)
    if not found:
        print("Фрагменты по этому вопросу не найдены.")
        return

    for number, fragment in enumerate(found, start=1):
        print(f"\n{number}. {fragment.source}\n{fragment.heading}\n\n{fragment.text}")


if __name__ == "__main__":
    main()
