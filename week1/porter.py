"""Week 1 — Porter stemmer, written from Porter (1980) so no library does the stemming (slide 11).

A word is a sequence of consonant (c) and vowel (v) groups: [C](VC)^m[V]. The measure m counts
the VC pairs, and most suffix rules only fire when the remaining stem has a large enough m.

    Step 1  plurals and -ed/-ing        caresses -> caress, hopping -> hop
    Step 2  double suffixes             relational -> relate
    Step 3  -ic-, -full, -ness ...      hopeful -> hope
    Step 4  remove -ant, -ence ...      revival -> reviv   (needs m > 1)
    Step 5  tidy a final -e or -ll      probate -> probat

The small refinements on top of the 1980 paper (irregular words, -ied, -logi, -fulli) follow the
widely used NLTK variant, so results stay comparable with Weeks 1–5.
"""
IRREGULAR = {"skies": "sky", "sky": "sky", "dying": "die", "lying": "lie", "tying": "tie", "news": "news",
             "innings": "inning", "inning": "inning", "outings": "outing", "outing": "outing",
             "cannings": "canning", "canning": "canning", "howe": "howe", "proceed": "proceed",
             "exceed": "exceed", "succeed": "succeed"}
VOWELS = frozenset("aeiou")


def is_consonant(word, i):
    if word[i] in VOWELS:
        return False
    if word[i] == "y":
        return i == 0 or not is_consonant(word, i - 1)
    return True


def measure(stem):
    """m in [C](VC)^m[V]."""
    return "".join("c" if is_consonant(stem, i) else "v" for i in range(len(stem))).count("vc")


def contains_vowel(stem):
    return any(not is_consonant(stem, i) for i in range(len(stem)))


def ends_double_consonant(word):
    return len(word) >= 2 and word[-1] == word[-2] and is_consonant(word, len(word) - 1)


def ends_cvc(word):
    n = len(word)
    if n >= 3:
        return (is_consonant(word, n - 3) and not is_consonant(word, n - 2) and is_consonant(word, n - 1)
                and word[-1] not in "wxy")
    return n == 2 and not is_consonant(word, 0) and is_consonant(word, 1)


def positive(stem):
    return measure(stem) > 0


def above_one(stem):
    return measure(stem) > 1


def apply_rules(word, rules):
    """The first rule whose suffix matches decides, whether or not its condition then holds."""
    for suffix, replacement, condition in rules:
        if suffix == "*d":                     # any double consonant
            if ends_double_consonant(word):
                stem = word[:-2]
                return stem + replacement if condition is None or condition(stem) else word
        elif word.endswith(suffix):
            stem = word[:-len(suffix)] if suffix else word
            return stem + replacement if condition is None or condition(stem) else word
    return word


def step1a(word):
    if word.endswith("ies") and len(word) == 4:
        return word[:-3] + "ie"
    return apply_rules(word, [("sses", "ss", None), ("ies", "i", None), ("ss", "ss", None), ("s", "", None)])


def step1b(word):
    if word.endswith("ied"):
        return word[:-3] + ("ie" if len(word) == 4 else "i")
    if word.endswith("eed"):
        return word[:-3] + "ee" if measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix) and contains_vowel(word[:-len(suffix)]):
            stem = word[:-len(suffix)]
            last = stem[-1]
            return apply_rules(stem, [("at", "ate", None), ("bl", "ble", None), ("iz", "ize", None),
                                      ("*d", last, lambda s: last not in "lsz"),
                                      ("", "e", lambda s: measure(s) == 1 and ends_cvc(s))])
    return word


def step1c(word):
    return apply_rules(word, [("y", "i", lambda s: len(s) > 1 and is_consonant(s, len(s) - 1))])


def step2(word):
    if word.endswith("alli") and positive(word[:-4]):
        return step2(word[:-4] + "al")
    return apply_rules(word, [
        ("ational", "ate", positive), ("tional", "tion", positive), ("enci", "ence", positive),
        ("anci", "ance", positive), ("izer", "ize", positive), ("bli", "ble", positive),
        ("alli", "al", positive), ("entli", "ent", positive), ("eli", "e", positive), ("ousli", "ous", positive),
        ("ization", "ize", positive), ("ation", "ate", positive), ("ator", "ate", positive),
        ("alism", "al", positive), ("iveness", "ive", positive), ("fulness", "ful", positive),
        ("ousness", "ous", positive), ("aliti", "al", positive), ("iviti", "ive", positive),
        ("biliti", "ble", positive), ("fulli", "ful", positive),
        ("logi", "log", lambda s: positive(word[:-3])),
    ])


def step3(word):
    return apply_rules(word, [("icate", "ic", positive), ("ative", "", positive), ("alize", "al", positive),
                              ("iciti", "ic", positive), ("ical", "ic", positive), ("ful", "", positive),
                              ("ness", "", positive)])


def step4(word):
    return apply_rules(word, [("al", "", above_one), ("ance", "", above_one), ("ence", "", above_one),
                              ("er", "", above_one), ("ic", "", above_one), ("able", "", above_one),
                              ("ible", "", above_one), ("ant", "", above_one), ("ement", "", above_one),
                              ("ment", "", above_one), ("ent", "", above_one),
                              ("ion", "", lambda s: measure(s) > 1 and s[-1] in "st"),
                              ("ou", "", above_one), ("ism", "", above_one), ("ate", "", above_one),
                              ("iti", "", above_one), ("ous", "", above_one), ("ive", "", above_one),
                              ("ize", "", above_one)])


def step5a(word):
    if word.endswith("e"):
        stem, m = word[:-1], measure(word[:-1])
        if m > 1 or (m == 1 and not ends_cvc(stem)):
            return stem
    return word


def step5b(word):
    return word[:-1] if word.endswith("ll") and measure(word[:-1]) > 1 else word


def stem_word(word):
    word = word.lower()
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) <= 2:
        return word
    for step in (step1a, step1b, step1c, step2, step3, step4, step5a, step5b):
        word = step(word)
    return word
