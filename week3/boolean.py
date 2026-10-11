"""Week 3 — Boolean model (slides 5–9).

Rel(D, Q) = 1 if D satisfies the logical expression Q, else 0. No ranking.

Supports AND, OR, NOT, parentheses, and implicit AND between adjacent terms.
Precedence: NOT > AND > OR.  Example:  iot AND intrusion AND (edge OR deployment)
"""
import re

from week1.preprocess import preprocess

_TOKEN = re.compile(r"\(|\)|[^\s()]+")
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}


class BooleanQueryError(ValueError):
    pass


def _lex(query):
    """Split into operators/parens/terms and insert implicit ANDs."""
    raw = _TOKEN.findall(query)
    out = []
    for tok in raw:
        kind = tok.upper() if tok.upper() in PRECEDENCE else tok
        is_operand_start = kind not in ("AND", "OR", ")")
        prev_ends_operand = out and out[-1] not in ("AND", "OR", "NOT", "(")
        if is_operand_start and prev_ends_operand:
            out.append("AND")
        out.append(kind)
    return out


def to_postfix(query):
    """Shunting-yard: infix tokens -> postfix (reverse Polish) list."""
    output, stack = [], []
    for tok in _lex(query):
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise BooleanQueryError("Unbalanced ')'")
            stack.pop()
        elif tok in PRECEDENCE:
            # NOT is right-associative unary; AND/OR are left-associative.
            while (stack and stack[-1] in PRECEDENCE and tok != "NOT"
                   and PRECEDENCE[stack[-1]] >= PRECEDENCE[tok]):
                output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(("TERM", tok))
    while stack:
        if stack[-1] == "(":
            raise BooleanQueryError("Unbalanced '('")
        output.append(stack.pop())
    return output


class BooleanModel:
    def __init__(self, index):
        self.index = index
        self.universe = set(index.doc_ids)

    def term_set(self, word):
        stems = preprocess(word)
        if not stems:            # a stop-word: neutral element, matches everything
            return set(self.universe)
        docs = set(self.index.lookup(stems[0]))
        for s in stems[1:]:      # hyphenated words like "pre-processing" become several stems
            docs &= set(self.index.lookup(s))
        return docs

    def search(self, query):
        """Set of doc_ids satisfying the query."""
        stack = []
        for tok in to_postfix(query):
            if isinstance(tok, tuple):
                stack.append(self.term_set(tok[1]))
            elif tok == "NOT":
                if not stack:
                    raise BooleanQueryError("NOT needs an operand")
                stack.append(self.universe - stack.pop())
            else:
                if len(stack) < 2:
                    raise BooleanQueryError(f"{tok} needs two operands")
                b, a = stack.pop(), stack.pop()
                stack.append(a & b if tok == "AND" else a | b)
        if len(stack) != 1:
            raise BooleanQueryError("Malformed query")
        return stack[0]

    def rel(self, query, doc_id):
        """Rel(D, Q) ∈ {0, 1}"""
        return int(doc_id in self.search(query))
