"""Russian suffix stemming using the published Snowball algorithm.

Algorithm specification: https://snowballstem.org/algorithms/russian/stemmer.html
This implementation uses no third-party packages; it is not a synonym dictionary.
"""

import re
from functools import lru_cache

VOWELS = frozenset("аеиоуыэюя")


def endings(text, conditional=""):
    return sorted([(suffix, False) for suffix in text.split()]
                  + [(suffix, True) for suffix in conditional.split()],
                  key=lambda item: len(item[0]), reverse=True)


GERUND = endings("ив ивши ившись ыв ывши ывшись", "в вши вшись")
ADJECTIVE = endings("ее ие ые ое ими ыми ей ий ый ой ем им ым ом его ого ему ому их ых ую юю ая яя ою ею")
PARTICIPLE = endings("ивш ывш ующ", "ем нн вш ющ щ")
REFLEXIVE = endings("ся сь")
VERB = endings("ила ыла ена ейте уйте ите или ыли ей уй ил ыл им ым ен ило ыло ено ят ует уют ит ыт ены ить ыть ишь ую ю",
               "ла на ете йте ли й л ем н ло но ет ют ны ть ешь нно")
NOUN = endings("а ев ов ие ье е иями ями ами еи ии и ией ей ой ий й иям ям ием ем ам ом о у ах иях ях ы ь ию ью ю ия ья я")


def remove(word, suffixes, start):
    for suffix, conditional in suffixes:
        pos = len(word) - len(suffix)
        if pos < start or not word.endswith(suffix):
            continue
        if conditional and (pos <= start or word[pos - 1] not in "ая"):
            continue
        return word[:pos], True
    return word, False


def region(word, start):
    for i in range(start + 1, len(word)):
        if word[i - 1] in VOWELS and word[i] not in VOWELS:
            return i + 1
    return len(word)


@lru_cache(maxsize=32768)
def stem(word):
    word = word.casefold().replace("ё", "е")
    if not re.fullmatch("[а-я]+", word):
        return word
    rv = next((i + 1 for i, letter in enumerate(word) if letter in VOWELS), len(word))
    r2 = region(word, region(word, 0))
    word, removed = remove(word, GERUND, rv)
    if not removed:
        word, _ = remove(word, REFLEXIVE, rv)
        word, adjective = remove(word, ADJECTIVE, rv)
        if adjective:
            word, _ = remove(word, PARTICIPLE, rv)
        else:
            word, verb = remove(word, VERB, rv)
            if not verb:
                word, _ = remove(word, NOUN, rv)
    word, _ = remove(word, endings("и"), rv)
    word, _ = remove(word, endings("ость ост"), max(rv, r2))
    if word.endswith("нн") and len(word) - 2 >= rv:
        word = word[:-1]
    else:
        word, superlative = remove(word, endings("ейше ейш"), rv)
        if superlative:
            if word.endswith("нн") and len(word) - 2 >= rv:
                word = word[:-1]
        else:
            word, _ = remove(word, endings("ь"), rv)
    return word
