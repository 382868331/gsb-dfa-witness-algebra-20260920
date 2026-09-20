"""演示与测试共用的样例自动机构造。"""

from dfalib import DFA


def ends_with(suffix, alphabet):
    """接受以 suffix 结尾的串的最小完备 DFA（KMP 自动机）。"""
    n = len(suffix)
    states = list(range(n + 1))
    triples = []
    for i in states:
        for tok in alphabet:
            s2 = suffix[:i] + tok
            k = min(n, len(s2))
            while k > 0 and not s2.endswith(suffix[:k]):
                k -= 1
            triples.append((i, tok, k))
    return DFA(states, list(alphabet), 0, [n], triples)


def mod_counter(k, residue=0, token="a", alphabet=("a",)):
    """接受 token 计数 ≡ residue (mod k) 的串；其余 token 进汇点。"""
    states = list(range(k))
    triples = [(i, token, (i + 1) % k) for i in states]
    return DFA(states, list(alphabet), 0, [residue], triples)


def chain(n, token="a"):
    """长链 0 -> 1 -> ... -> n-1，仅末端接受。"""
    states = list(range(n))
    triples = [(i, token, i + 1) for i in range(n - 1)]
    return DFA(states, [token], 0, [n - 1], triples)
