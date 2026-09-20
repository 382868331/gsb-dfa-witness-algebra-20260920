"""测试共享辅助：短串穷举、区分后缀、随机小机。"""

import random
from collections import deque

from dfalib import DFA


def all_strings(alphabet, max_len):
    """按 (长度, 字典序) 枚举所有长度 <= max_len 的 token 序列。"""
    result = [[]]
    frontier = [[]]
    for _ in range(max_len):
        frontier = [w + [t] for w in frontier for t in alphabet]
        result.extend(frontier)
    return result


def random_dfa(rng: random.Random, alphabet, n_states):
    """固定种子随机小机：转移以 0.8 概率存在（制造缺失转移）。"""
    states = list(range(n_states))
    accepting = [s for s in states if rng.random() < 0.35]
    triples = []
    for s in states:
        for tok in alphabet:
            if rng.random() < 0.8:
                triples.append((s, tok, rng.randrange(n_states)))
    return DFA(states, list(alphabet), 0, accepting, triples)


def distinguishing_suffix(dfa, p, q):
    """在自身乘积图上 BFS，找区分状态 p、q 的最短后缀；不存在返回 None。

    隐式汇点用 None 表示，缺失转移进入汇点。
    """
    def acc(s):
        return s is not None and s in dfa.accepting

    start = (p, q)
    parent = {start: None}
    dq = deque([start])
    while dq:
        x, y = dq.popleft()
        if acc(x) != acc(y):
            path = []
            n = (x, y)
            while parent[n] is not None:
                prev, tok = parent[n]
                path.append(tok)
                n = prev
            path.reverse()
            return path
        for tok in dfa.alphabet:
            nx = None if x is None else dfa.transitions.get(x, {}).get(tok)
            ny = None if y is None else dfa.transitions.get(y, {}).get(tok)
            nxt = (nx, ny)
            if nxt not in parent:
                parent[nxt] = ((x, y), tok)
                dq.append(nxt)
    return None
