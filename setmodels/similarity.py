"""Week 6 (set-theoretic models) — Word similarity for partial matching (Ratcliff–Obershelp, 1988).

    sim(a, b) = 2M / T      T = |a| + |b|,  M = characters in matching blocks

Find the longest common block, then repeat on the text left of it and right of it.
"""


def longest_block(a, b):
    """(start in a, start in b, size) of the longest common substring, earliest first."""
    best = (0, 0, 0)
    previous = [0] * (len(b) + 1)
    for i, ch in enumerate(a):
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b):
            if ch == other:
                current[j + 1] = previous[j] + 1
                if current[j + 1] > best[2]:
                    best = (i - current[j + 1] + 1, j - current[j + 1] + 1, current[j + 1])
        previous = current
    return best


def matching_characters(a, b):
    if not a or not b:
        return 0
    i, j, size = longest_block(a, b)
    if not size:
        return 0
    return size + matching_characters(a[:i], b[:j]) + matching_characters(a[i + size:], b[j + size:])


def similarity(a, b):
    """1.0 for identical words, 0.0 for words sharing no characters."""
    total = len(a) + len(b)
    if not total:
        return 1.0
    return 2.0 * matching_characters(a, b) / total
