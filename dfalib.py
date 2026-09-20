"""DFA 语言运算库（不解析正则表达式）。

对显式给出的确定有限自动机做交、并、差、补、最小化，
以及等价/包含判断（失败时返回最短见证 token 列表）。

语义约定：
- 字母表为显式给出的字符串 token 集合（最多 8 个），内部按字典序排序；
- 状态 id 只能是整数；未知状态引用、重复字母、重复转移一律拒绝；
- 转移表可以不完整，缺失的转移进入隐式拒绝汇点；
- 所有二元运算先把双方字母表统一为并集（字典序），
  原机未声明的 token 同样进入汇点；
- 外部导入的原始 DFA 最多 80 个状态；运算结果允许更多，
  单次运算最多探索 DEFAULT_STATE_LIMIT 个可达状态，
  实际探索达到上限后仍需新状态时抛出 StateLimitError，
  不按笛卡尔积总大小预先拒绝稀疏可达的乘积。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

MAX_ALPHABET = 8
DEFAULT_STATE_LIMIT = 10_000
DEFAULT_IMPORT_MAX_STATES = 80


class DFAError(Exception):
    """本库所有错误的基类。"""


class ValidationError(DFAError):
    """输入 DFA 不合法（未知状态引用、重复字母、重复转移等）。"""


class StateLimitError(DFAError):
    """单次运算的可达状态探索达到上限后仍需新状态。"""

    def __init__(self, limit: int):
        super().__init__(
            f"状态上限已触发：探索达到 {limit} 个可达状态后仍需新状态"
        )
        self.limit = limit


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class DFA:
    """确定有限自动机。转移表可不完整，缺失转移进入隐式拒绝汇点。

    transitions 为 (src, token, dst) 三元组的可迭代对象，
    这样重复的 (src, token) 可以被检测并拒绝。
    构造后请将 transitions 属性视为只读。
    """

    __slots__ = ("states", "alphabet", "start", "accepting", "transitions")

    def __init__(self, states, alphabet, start, accepting, transitions,
                 *, max_states=DEFAULT_IMPORT_MAX_STATES):
        state_list = list(states)
        for s in state_list:
            if not _is_int(s):
                raise ValidationError(f"状态 id 必须是整数: {s!r}")
        state_set = frozenset(state_list)
        if max_states is not None and len(state_set) > max_states:
            raise ValidationError(
                f"状态数 {len(state_set)} 超过导入上限 {max_states}"
            )

        alpha_list = list(alphabet)
        for tok in alpha_list:
            if not isinstance(tok, str):
                raise ValidationError(f"字母表 token 必须是字符串: {tok!r}")
        if len(set(alpha_list)) != len(alpha_list):
            raise ValidationError("字母表存在重复 token")
        if len(alpha_list) > MAX_ALPHABET:
            raise ValidationError(
                f"字母表大小 {len(alpha_list)} 超过上限 {MAX_ALPHABET}"
            )
        alpha_set = frozenset(alpha_list)

        if not _is_int(start):
            raise ValidationError(f"初态 id 必须是整数: {start!r}")
        if start not in state_set:
            raise ValidationError(f"初态 {start} 不在状态集合中")

        acc = set()
        for s in accepting:
            if not _is_int(s):
                raise ValidationError(f"接受态 id 必须是整数: {s!r}")
            if s not in state_set:
                raise ValidationError(f"接受态引用了未知状态: {s}")
            acc.add(s)

        table: dict[int, dict[str, int]] = {}
        for triple in transitions:
            try:
                src, tok, dst = triple
            except (TypeError, ValueError):
                raise ValidationError(f"转移必须是 (src, token, dst) 三元组: {triple!r}")
            if not _is_int(src) or src not in state_set:
                raise ValidationError(f"转移源引用了未知状态: {src!r}")
            if not _is_int(dst) or dst not in state_set:
                raise ValidationError(f"转移目标引用了未知状态: {dst!r}")
            if tok not in alpha_set:
                raise ValidationError(f"转移使用了字母表外的 token: {tok!r}")
            row = table.setdefault(src, {})
            if tok in row:
                raise ValidationError(f"重复转移: ({src}, {tok!r})")
            row[tok] = dst

        self.states = state_set
        self.alphabet = tuple(sorted(alpha_list))
        self.start = start
        self.accepting = frozenset(acc)
        self.transitions = table

    def __repr__(self):
        return (f"DFA(states={len(self.states)}, alphabet={self.alphabet!r}, "
                f"start={self.start}, accepting={sorted(self.accepting)})")

    def accepts(self, tokens) -> bool:
        """按隐式汇点语义运行 token 序列；未声明的 token 同样进入汇点。"""
        s = self.start
        for tok in tokens:
            row = self.transitions.get(s)
            if row is None:
                return False
            s = row.get(tok)
            if s is None:
                return False
        return s in self.accepting


def _step(dfa: DFA, s, tok):
    """内部单步；None 表示隐式汇点，汇点自环。"""
    if s is None:
        return None
    return dfa.transitions.get(s, {}).get(tok)


def _acc(dfa: DFA, s) -> bool:
    return s is not None and s in dfa.accepting


def _reachable_states(dfa: DFA) -> set:
    seen = {dfa.start}
    dq = deque([dfa.start])
    while dq:
        s = dq.popleft()
        for tok in dfa.alphabet:
            t = dfa.transitions.get(s, {}).get(tok)
            if t is not None and t not in seen:
                seen.add(t)
                dq.append(t)
    return seen


def _union_alphabet(a: DFA, b: DFA) -> tuple:
    return tuple(sorted(set(a.alphabet) | set(b.alphabet)))


def _product_op(a: DFA, b: DFA, mode: str, limit: int) -> DFA:
    """交/并/差的可达乘积构造，按 (发现顺序) 重新编号为 0..n-1。"""
    alphabet = _union_alphabet(a, b)
    start = (a.start, b.start)
    ids = {start: 0}
    triples = []
    accepting = []
    dq = deque([start])
    while dq:
        node = dq.popleft()
        sa, sb = node
        fa, fb = _acc(a, sa), _acc(b, sb)
        if mode == "intersection":
            keep = fa and fb
        elif mode == "union":
            keep = fa or fb
        else:  # difference
            keep = fa and not fb
        if keep:
            accepting.append(ids[node])
        for tok in alphabet:
            nxt = (_step(a, sa, tok), _step(b, sb, tok))
            if nxt not in ids:
                if len(ids) >= limit:
                    raise StateLimitError(limit)
                ids[nxt] = len(ids)
                dq.append(nxt)
            triples.append((ids[node], tok, ids[nxt]))
    return DFA(range(len(ids)), alphabet, 0, accepting, triples, max_states=None)


def intersect(a: DFA, b: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> DFA:
    """交：接受 L(a) ∩ L(b)，字母表为双方并集。"""
    return _product_op(a, b, "intersection", limit)


def union(a: DFA, b: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> DFA:
    """并：接受 L(a) ∪ L(b)，字母表为双方并集。"""
    return _product_op(a, b, "union", limit)


def difference(a: DFA, b: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> DFA:
    """差：接受 L(a) \\ L(b)，字母表为双方并集。"""
    return _product_op(a, b, "difference", limit)


def complement(dfa: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> DFA:
    """补：相对于自身显式字母表的全部有限串。空字母表下只有空串。"""
    reachable = _reachable_states(dfa)
    need_sink = any(
        tok not in dfa.transitions.get(s, ())
        for s in reachable for tok in dfa.alphabet
    )
    n = len(reachable) + (1 if need_sink else 0)
    if n > limit:
        raise StateLimitError(limit)
    states = set(reachable)
    sink_id = None
    if need_sink:
        sink_id = max(dfa.states) + 1
        states.add(sink_id)
    triples = []
    for s in reachable:
        for tok in dfa.alphabet:
            t = dfa.transitions.get(s, {}).get(tok)
            triples.append((s, tok, t if t is not None else sink_id))
    if need_sink:
        for tok in dfa.alphabet:
            triples.append((sink_id, tok, sink_id))
    accepting = [s for s in reachable if s not in dfa.accepting]
    if need_sink:
        # 隐式汇点在原机中是拒绝态，补机中必须翻转为接受态
        accepting.append(sink_id)
    return DFA(states, dfa.alphabet, dfa.start, accepting, triples,
               max_states=None)


def _hopcroft(alphabet, states, accepting, trans):
    """Hopcroft 分割细化。trans 必须是 states 上完备的转移表。

    返回 (blocks, block_of)。
    """
    acc = set(accepting) & states
    non = set(states) - acc
    blocks = [b for b in (acc, non) if b]
    block_of = {}
    for i, b in enumerate(blocks):
        for s in b:
            block_of[s] = i
    pred = {tok: {s: [] for s in states} for tok in alphabet}
    for s in states:
        for tok in alphabet:
            pred[tok][trans[s][tok]].append(s)
    W = deque(range(len(blocks)))
    in_W = set(W)
    while W:
        a_idx = W.popleft()
        in_W.discard(a_idx)
        A = blocks[a_idx]
        for tok in alphabet:
            X = set()
            for s in A:
                X.update(pred[tok][s])
            affected = {}
            for q in X:
                affected.setdefault(block_of[q], []).append(q)
            for bi, inter in affected.items():
                B = blocks[bi]
                if len(inter) == len(B):
                    continue
                inter_set = set(inter)
                rest = [s for s in B if s not in inter_set]
                blocks[bi] = inter
                new_idx = len(blocks)
                blocks.append(rest)
                for s in inter:
                    block_of[s] = bi
                for s in rest:
                    block_of[s] = new_idx
                if bi in in_W:
                    in_W.discard(bi)
                    W.append(bi)
                    W.append(new_idx)
                    in_W.add(bi)
                    in_W.add(new_idx)
                else:
                    smaller = bi if len(inter) <= len(rest) else new_idx
                    W.append(smaller)
                    in_W.add(smaller)
    return blocks, block_of


@dataclass(frozen=True)
class Minimized:
    """最小化结果。

    dfa: 真正最小的完备 DFA，从初态 0 开始按 token 顺序 BFS 重新编号；
    state_map: 原可达状态 -> 最小化状态 id（不可达状态不出现）；
    sink_state: 原机隐式汇点可达时，它在最小化机中的状态 id，否则为 None。
    """

    dfa: DFA
    state_map: dict
    sink_state: int | None


def minimize(dfa: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> Minimized:
    """分割细化最小化：只保留可达状态，产出真正最小的完备 DFA。"""
    reachable = _reachable_states(dfa)
    need_sink = any(
        tok not in dfa.transitions.get(s, ())
        for s in reachable for tok in dfa.alphabet
    )
    work = set(reachable)
    sink_id = None
    if need_sink:
        sink_id = max(dfa.states) + 1
        work.add(sink_id)
    if len(work) > limit:
        raise StateLimitError(limit)

    trans = {}
    for s in work:
        row = {}
        for tok in dfa.alphabet:
            if s == sink_id:
                row[tok] = sink_id
            else:
                t = dfa.transitions.get(s, {}).get(tok)
                row[tok] = t if t is not None else sink_id
        trans[s] = row
    accepting_work = set(dfa.accepting) & reachable

    blocks, block_of = _hopcroft(dfa.alphabet, work, accepting_work, trans)
    rep = [next(iter(b)) for b in blocks]

    # 从初态所在块开始按 token 顺序 BFS 重新编号
    start_block = block_of[dfa.start]
    new_id = {start_block: 0}
    order = [start_block]
    dq = deque([start_block])
    while dq:
        b = dq.popleft()
        for tok in dfa.alphabet:
            nb = block_of[trans[rep[b]][tok]]
            if nb not in new_id:
                new_id[nb] = len(new_id)
                order.append(nb)
                dq.append(nb)

    triples = []
    acc_new = []
    for b in order:
        r = rep[b]
        if r in accepting_work:
            acc_new.append(new_id[b])
        for tok in dfa.alphabet:
            triples.append((new_id[b], tok, new_id[block_of[trans[r][tok]]]))
    min_dfa = DFA(range(len(blocks)), dfa.alphabet, 0, acc_new, triples,
                  max_states=None)
    state_map = {s: new_id[block_of[s]] for s in reachable}
    sink_state = new_id[block_of[sink_id]] if need_sink else None
    return Minimized(min_dfa, state_map, sink_state)


def _product_witness(a: DFA, b: DFA, is_target, limit: int):
    """在可达乘积图上 BFS，返回满足 is_target(acc_a, acc_b) 的最短路径。

    同长度按 token 序列字典序最小（BFS 层序 + token 字典序扩展）。
    找不到返回 None；空列表是合法结果（初态对即为目标）。
    """
    alphabet = _union_alphabet(a, b)
    start = (a.start, b.start)
    parent = {start: None}
    dq = deque([start])
    while dq:
        node = dq.popleft()
        sa, sb = node
        if is_target(_acc(a, sa), _acc(b, sb)):
            path = []
            n = node
            while parent[n] is not None:
                p, tok = parent[n]
                path.append(tok)
                n = p
            path.reverse()
            return path
        for tok in alphabet:
            nxt = (_step(a, sa, tok), _step(b, sb, tok))
            if nxt not in parent:
                if len(parent) >= limit:
                    raise StateLimitError(limit)
                parent[nxt] = (node, tok)
                dq.append(nxt)
    return None


@dataclass(frozen=True)
class EquivalenceResult:
    """等价判断结果。

    equivalent 为 False 时 witness 是最短见证 token 列表
    （空列表也是合法见证，与 equivalent=True 时的 None 不同），
    accepted_by_a / accepted_by_b 是该见证在双方上的实际接受性。
    """

    equivalent: bool
    witness: list | None
    accepted_by_a: bool | None
    accepted_by_b: bool | None


@dataclass(frozen=True)
class InclusionResult:
    """包含判断结果：L(a) ⊆ L(b) 是否成立。

    contained 为 False 时 witness 是属于 L(a) \\ L(b) 的最短见证。
    """

    contained: bool
    witness: list | None
    accepted_by_a: bool | None
    accepted_by_b: bool | None


def check_equivalent(a: DFA, b: DFA, *,
                     limit: int = DEFAULT_STATE_LIMIT) -> EquivalenceResult:
    """判断 L(a) == L(b)；失败时返回最短见证并在双方上核验。"""
    w = _product_witness(a, b, lambda x, y: x != y, limit)
    if w is None:
        return EquivalenceResult(True, None, None, None)
    aa, bb = a.accepts(w), b.accepts(w)
    if aa == bb:  # 乘积搜索与单机运行不一致，属于内部错误
        raise DFAError(f"见证核验失败: {w!r} 在双方接受性相同")
    return EquivalenceResult(False, w, aa, bb)


def check_included(a: DFA, b: DFA, *,
                   limit: int = DEFAULT_STATE_LIMIT) -> InclusionResult:
    """判断 L(a) ⊆ L(b)；失败时返回 L(a) \\ L(b) 的最短见证并核验。"""
    w = _product_witness(a, b, lambda x, y: x and not y, limit)
    if w is None:
        return InclusionResult(True, None, None, None)
    aa, bb = a.accepts(w), b.accepts(w)
    if not (aa and not bb):
        raise DFAError(f"见证核验失败: {w!r} 不属于 L(a) \\ L(b)")
    return InclusionResult(False, w, aa, bb)


def are_equivalent(a: DFA, b: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> bool:
    return check_equivalent(a, b, limit=limit).equivalent


def is_included(a: DFA, b: DFA, *, limit: int = DEFAULT_STATE_LIMIT) -> bool:
    return check_included(a, b, limit=limit).contained
