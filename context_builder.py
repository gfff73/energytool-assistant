"""Assemble bounded evidence from sections of the documents retrieval selected.

The initial hits still decide which documents are relevant. Expansion never
introduces a different document or turns an empty search into a positive hit.
Whole sections preserve table rows, qualifications and nearby supporting facts.
"""

from collections import OrderedDict

from search_kb import Fragment


def context_cost(fragment):
    # Upper bound on the numbered heading and separators used by Consultant.
    # This is a character budget, not a claim about the model's tokenization.
    return len(fragment.heading) + len(fragment.text) + 16


class ContextBuilder:
    def __init__(self, index, max_characters=16000):
        self.index = index
        self.max_characters = max_characters
        grouped = OrderedDict()
        for fragment in index.fragments:
            grouped.setdefault((fragment.source, fragment.heading), []).append(fragment.text)
        self.sections = {
            key: Fragment(key[0], key[1], "\n\n".join(parts))
            for key, parts in grouped.items()
        }

    def build(self, question, anchors):
        if not anchors:
            return [], {"characters": 0, "sections": 0, "documents": 0, "omitted_sections": 0}

        selected = OrderedDict()
        remaining = self.max_characters

        def add(fragment):
            nonlocal remaining
            key = (fragment.source, fragment.heading)
            if key in selected:
                return False
            cost = context_cost(fragment)
            if cost > remaining:
                return False
            selected[key] = fragment
            remaining -= cost
            return True

        # Admit each document together with its guardrails. A tight budget must
        # not drop its limitations and rely on an unbudgeted later append.
        guardrails = {key: section for key, section in self.sections.items()
                      if key[1].endswith("Чего утверждать нельзя")}
        pending_guards = OrderedDict()
        for anchor in anchors:
            key = (anchor.source, anchor.heading)
            if key in selected or key in pending_guards:
                continue
            evidence = guardrails.get(key, anchor)
            missing_guards = {k: section for k, section in guardrails.items()
                              if k[0] == anchor.source and k != key
                              and k not in selected and k not in pending_guards}
            required = context_cost(evidence) + sum(context_cost(f) for f in missing_guards.values())
            if required <= remaining:
                selected[key] = evidence
                pending_guards.update(missing_guards)
                remaining -= required
        selected.update(pending_guards)
        sources = {key[0] for key in selected}

        # Upgrade the selected passages to their full original sections. Do not
        # slice a section or table to make it fit; keep the initial hit instead.
        for key, fragment in list(selected.items()):
            complete = self.sections[key]
            extra = context_cost(complete) - context_cost(fragment)
            if extra <= remaining:
                selected[key] = complete
                remaining -= extra

        # A small subsection may rely on the introduction of its parent. Avoid
        # promoting the document title itself to an unrelated whole-document hit.
        for anchor in anchors:
            if anchor.source not in sources:
                continue
            parts = anchor.heading.split(" / ")
            for depth in range(len(parts) - 1, 1, -1):
                parent = self.sections.get((anchor.source, " / ".join(parts[:depth])))
                if parent is not None:
                    add(parent)

        # Look beyond the narrow initial cutoff, but only within documents that
        # already passed retrieval. A reranker or dense hit remains an anchor.
        scores = {}
        for hit in self.index.rank(question, limit=len(self.index.fragments), filter_relevance=False):
            key = (hit.fragment.source, hit.fragment.heading)
            if key[0] in sources:
                scores[key] = max(scores.get(key, 0.0), hit.score)
        candidates = [key for key in self.sections if key[0] in sources and key not in selected]
        candidates.sort(key=lambda key: -scores.get(key, 0.0))
        for key in candidates:
            add(self.sections[key])

        result = list(selected.values())
        return result, {
            "characters": self.max_characters - remaining,
            "budget_characters": self.max_characters,
            "sections": len(result),
            "documents": len({fragment.source for fragment in result}),
            "omitted_sections": sum(key[0] in sources and key not in selected for key in self.sections),
        }
