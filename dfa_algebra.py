"""DFA 语言代数库（不解析正则表达式）。

接口概览
--------
- ``DFA(states, alphabet, start, accepting, transitions)``：显式、可能不完整的 DFA。
  转移缺失视为进入隐式拒绝汇点（内部以哨兵 ``SINK`` 表示）。
- 语言运算：``intersect`` / ``union`` / ``difference`` / ``complement``。
- ``minimize``：Hopcroft 分割细化，只保留可达状态，输出从 0 开始、按 token
  字典序 BFS 重新编号的完整最小 DFA，并返回原可达状态（含隐式汇点）到最小
  状态的映射。
- 判定：``equivalent`` / ``subset``（布尔）以及 ``witness_inequivalent`` /
  ``witness_not_subset``（失败时返回最短、同长度字典序最小的见证 token 列表；
  无见证返回 ``None``；空列表 ``[]`` 是合法见证 ε）。
- ``run(dfa, word)``：逐步执行，返回是否接受。

输入限制（见模块常量）：外部原始 DFA 至多 ``IMPORTED_STATE_LIMIT`` 个状态；
显式字母表至多 ``ALPHABET_MAX`` 个字符串 token；单次运算可达状态至多
``MAX_REACHABLE_STATES`` 个，探索中需要第 10001 个新状态时抛
``DFALimitError``，不预先按笛卡尔积总大小拒绝。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

ALPHABET_MAX = 8
IMPORTED_STATE_LIMIT = 80
MAX_REACHABLE_STATES = 10000

#: 隐式拒绝汇点的内部哨兵 id。它不是合法的外部状态 id。
SINK = -1


class DFAError(ValueError):
    """DFA 结构非法（未知引用、重复字母/转移、类型错误等）。"""


class DFALimitError(DFAError):
    """运算探索中可达状态数超过 ``MAX_REACHABLE_STATES``。"""


# ---------------------------------------------------------------------------
# DFA 定义
# ---------------------------------------------------------------------------


class DFA:
    """显式 DFA。

    参数
    ----
    states:
        整数状态 id 的可迭代对象（显式声明的状态全集）。
    alphabet:
        字符串 token 的可迭代对象，按字典序排序保存；不允许重复，至多 8 个；
        允许空字母表。
    start:
        初态 id，必须在 states 中。
    accepting:
        接受态 id 集合，必须是 states 的子集。
    transitions:
        不完整转移表，接受两种形式：

        * ``{(src, token): dst, ...}`` 映射；
        * ``[(src, token, dst), ...]`` 三元组可迭代对象（此形式会检测重复转移）。

        src/dst 必须是 states 中的整数，token 必须属于 alphabet；
        未声明的 ``(src, token)`` 视为进入隐式拒绝汇点。
    """

    __slots__ = (
        "states",
        "alphabet",
        "start",
        "accepting",
        "transitions",
        "_idx",
        "_rows",
        "_acc",
    )

    def __init__(
        self,
        states: Iterable[int],
        alphabet: Iterable[str],
        start: int,
        accepting: Iterable[int],
        transitions: Mapping[tuple[int, str], int] | Iterable[tuple[int, str, int]],
    ) -> None:
        state_list = list(states)
        for s in state_list:
            if not _is_plain_int(s):
                raise DFAError(f"状态 id 必须是整数: {s!r}")
        if len(state_list) != len(set(state_list)):
            raise DFAError("states 中存在重复状态 id")
        if not state_list:
            raise DFAError("states 不能为空")
        if len(state_list) > IMPORTED_STATE_LIMIT:
            raise DFAError(
                f"外部导入 DFA 至多 {IMPORTED_STATE_LIMIT} 个状态，"
                f"实际 {len(state_list)} 个（运算结果不受此限）"
            )
        state_set = frozenset(state_list)

        alpha_list = list(alphabet)
        for t in alpha_list:
            if not isinstance(t, str):
                raise DFAError(f"字母 token 必须是字符串: {t!r}")
        if len(alpha_list) != len(set(alpha_list)):
            raise DFAError("alphabet 中存在重复 token")
        if len(alpha_list) > ALPHABET_MAX:
            raise DFAError(
                f"字母表至多 {ALPHABET_MAX} 个 token，实际 {len(alpha_list)} 个"
            )
        alphabet_t = tuple(sorted(alpha_list))
        alpha_index = {t: j for j, t in enumerate(alphabet_t)}

        if not _is_plain_int(start):
            raise DFAError(f"初态必须是整数: {start!r}")
        if start not in state_set:
            raise DFAError(f"初态 {start} 未在 states 中声明")

        acc_list = list(accepting)
        for s in acc_list:
            if not _is_plain_int(s):
                raise DFAError(f"接受态 id 必须是整数: {s!r}")
            if s not in state_set:
                raise DFAError(f"接受态 {s} 未在 states 中声明")
        acc_set = frozenset(acc_list)

        triples: list[tuple[int, str, int]] = []
        if isinstance(transitions, Mapping):
            for key, dst in transitions.items():
                try:
                    src, tok = key
                except (TypeError, ValueError) as exc:
                    raise DFAError(f"非法转移键: {key!r}") from exc
                triples.append((src, tok, dst))
        else:
            seen: set[tuple[int, str]] = set()
            for item in transitions:
                try:
                    src, tok, dst = item  # type: ignore[misc]
                except (TypeError, ValueError) as exc:
                    raise DFAError(f"非法转移项: {item!r}") from exc
                key = (src, tok)
                if key in seen:
                    raise DFAError(f"重复转移: ({src!r}, {tok!r})")
                seen.add(key)
                triples.append((src, tok, dst))

        trans: dict[tuple[int, str], int] = {}
        for src, tok, dst in triples:
            if not _is_plain_int(src):
                raise DFAError(f"转移源状态必须是整数: {src!r}")
            if not _is_plain_int(dst):
                raise DFAError(f"转移目标状态必须是整数: {dst!r}")
            if src not in state_set:
                raise DFAError(f"转移源状态 {src} 未在 states 中声明")
            if dst not in state_set:
                raise DFAError(f"转移目标状态 {dst} 未在 states 中声明（缺失转移应省略而非指向汇点）")
            if not isinstance(tok, str) or tok not in alpha_index:
                raise DFAError(f"转移 token {tok!r} 不在字母表中")
            if (src, tok) in trans:
                raise DFAError(f"重复转移: ({src!r}, {tok!r})")
            trans[(src, tok)] = dst

        self.states: frozenset[int] = state_set
        self.alphabet: tuple[str, ...] = alphabet_t
        self.start: int = start
        self.accepting: frozenset[int] = acc_set
        self.transitions: dict[tuple[int, str], int] = trans

        # 完整化后的稠密转移表：最后一行是隐式汇点（全部指向汇点自身）。
        n = len(state_list)
        idx = {s: i for i, s in enumerate(state_list)}
        rows = [[SINK] * len(alphabet_t) for _ in range(n + 1)]
        for (src, tok), dst in trans.items():
            rows[idx[src]][alpha_index[tok]] = dst
        acc = [False] * n + [False]
        for s in acc_set:
            acc[idx[s]] = True
        self._idx = idx
        self._rows = rows
        self._acc = acc

    # ----- 内部使用的完整转移查询 -----

    def _go(self, state: int, sym_pos: int) -> int:
        """从（可能为 SINK 的）状态出发，走本机字母表位置 sym_pos 的 token。

        sym_pos 为 -1 表示统一字母表中的 token 未在本机声明，直接进入汇点。
        """
        if state == SINK or sym_pos < 0:
            return SINK
        return self._rows[self._idx[state]][sym_pos]

    def _accepts(self, state: int) -> bool:
        if state == SINK:
            return False
        return self._acc[self._idx[state]]

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"DFA(states={sorted(self.states)}, alphabet={list(self.alphabet)}, "
            f"start={self.start}, accepting={sorted(self.accepting)})"
        )


def _is_plain_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _unify(a: DFA, b: DFA) -> tuple[tuple[str, ...], list[int], list[int]]:
    """返回统一字母表（并集字典序）及双方各自的位置映射（缺失记 -1）。"""
    sigma = tuple(sorted(set(a.alphabet) | set(b.alphabet)))
    if len(sigma) > ALPHABET_MAX:
        raise DFAError(f"统一后字母表超过 {ALPHABET_MAX} 个 token")
    pa = {t: i for i, t in enumerate(a.alphabet)}
    pb = {t: i for i, t in enumerate(b.alphabet)}
    return sigma, [pa.get(t, -1) for t in sigma], [pb.get(t, -1) for t in sigma]


def _reachable_complete(
    d: DFA, sigma: Sequence[str], pos: Sequence[int]
) -> tuple[list[int], list[list[int]], list[bool]]:
    """在（可能扩大的）统一字母表上枚举完整化后的可达元素。

    返回 ``(order, rows, accept)``：

    * order[0] 是初态；元素为原状态 id 或 SINK；
    * rows[i][j] 给出 order 中目标元素的**位置**（不是 id）；
    * accept[i] 为该元素是否接受（汇点恒为 False）。
    """
    order = [d.start]
    at = {d.start: 0}
    rows: list[list[int]] = []
    accept: list[bool] = []
    q: deque[int] = deque([0])
    while q:
        i = q.popleft()
        x = order[i]
        row: list[int] = []
        for j, _tok in enumerate(sigma):
            y = d._go(x, pos[j])
            k = at.get(y)
            if k is None:
                if len(order) >= MAX_REACHABLE_STATES:
                    raise DFALimitError(
                        f"可达状态达到上限 {MAX_REACHABLE_STATES}，需要更多状态"
                    )
                k = len(order)
                at[y] = k
                order.append(y)
                q.append(k)
            row.append(k)
        rows.append(row)
        accept.append(d._accepts(x))
    return order, rows, accept


def _build_complete(
    sigma: Sequence[str],
    rows: list[list[int]],
    accept: Sequence[bool],
) -> DFA:
    """按给定完整表构造 DFA：状态即位置 0..m-1，初态 0，转移全声明。

    内部运算结果可达 10000 个状态，不受外部导入 80 状态限制，因此绕过
    构造器的输入校验直接装配（rows 必须是完整表）。
    """
    m = len(rows)
    d = object.__new__(DFA)
    d.states = frozenset(range(m))
    d.alphabet = tuple(sigma)
    d.start = 0
    d.accepting = frozenset(i for i, ok in enumerate(accept) if ok)
    d.transitions = {(i, sigma[j]): rows[i][j] for i in range(m) for j in range(len(sigma))}
    d._idx = {i: i for i in range(m)}
    # rows 已包含隐式汇点行（若探索中出现 SINK，它被当作普通位置）；
    # 完整表内不存在“缺失”，但为与 _go 的 SINK 分支兼容保留一行自环。
    d._rows = [list(r) for r in rows] + [[SINK] * len(sigma)]
    d._acc = [bool(accept[i]) for i in range(m)] + [False]
    return d


def _binary_op(
    a: DFA,
    b: DFA,
    accept_pair,
    op_name: str,
) -> DFA:
    """在统一字母表上对可达状态对做 BFS，按 accept_pair 判定接受。

    状态对在 BFS 中按整数编号（初态对为 0）；只有真正探索到新状态对时才
    检查上限，不预先使用笛卡尔积总大小。
    """
    sigma, pa, pb = _unify(a, b)
    start_pair = (a.start, b.start)
    pairs = [start_pair]
    pid = {start_pair: 0}
    rows: list[list[int]] = []
    accept: list[bool] = []
    q: deque[int] = deque([0])
    while q:
        i = q.popleft()
        x, y = pairs[i]
        row: list[int] = []
        for j in range(len(sigma)):
            npair = (a._go(x, pa[j]), b._go(y, pb[j]))
            k = pid.get(npair)
            if k is None:
                if len(pairs) >= MAX_REACHABLE_STATES:
                    raise DFALimitError(
                        f"{op_name}：可达乘积状态达到上限 {MAX_REACHABLE_STATES}"
                    )
                k = len(pairs)
                pid[npair] = k
                pairs.append(npair)
                q.append(k)
            row.append(k)
        rows.append(row)
        accept.append(bool(accept_pair(a, b, x, y)))
    return _build_complete(sigma, rows, accept)


# ---------------------------------------------------------------------------
# 语言运算
# ---------------------------------------------------------------------------


def intersect(a: DFA, b: DFA) -> DFA:
    """L(A) ∩ L(B)，字母表为双方并集（字典序）。"""
    return _binary_op(
        a, b, lambda ma, mb, x, y: ma._accepts(x) and mb._accepts(y), "intersect"
    )


def union(a: DFA, b: DFA) -> DFA:
    """L(A) ∪ L(B)，字母表为双方并集（字典序）。"""
    return _binary_op(
        a, b, lambda ma, mb, x, y: ma._accepts(x) or mb._accepts(y), "union"
    )


def difference(a: DFA, b: DFA) -> DFA:
    """L(A) \\ L(B)，字母表为双方并集（字典序）。"""
    return _binary_op(
        a, b, lambda ma, mb, x, y: ma._accepts(x) and not mb._accepts(y), "difference"
    )


def complement(a: DFA) -> DFA:
    """Σ* \\ L(A)，Σ 为 A 的显式字母表；缺失转移落入的汇点在补集中接受。

    空字母表时 Σ* = {ε}：仅按初态是否接受取反。
    """
    pos = list(range(len(a.alphabet)))
    order, rows, accept = _reachable_complete(a, a.alphabet, pos)
    return _build_complete(a.alphabet, rows, [not ok for ok in accept])


# ---------------------------------------------------------------------------
# 最小化（Hopcroft 分割细化）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Minimization:
    """:func:`minimize` 的结果。

    * ``dfa``：最小完整 DFA，可达状态从 0 开始按 token 顺序 BFS 编号；
    * ``mapping``：原机**可达**声明状态 id -> 最小状态 id；若隐式汇点可达，
      映射中还含键 ``SINK``（即 -1），并可通过 ``sink_state`` 取得；
    * ``sink_state``：汇点所在最小状态 id，不可达时为 None；
    * ``sink_reachable``：隐式汇点是否可达。
    """

    dfa: DFA
    mapping: dict[int, int]
    sink_state: int | None
    sink_reachable: bool


def minimize(d: DFA) -> Minimization:
    """只保留可达状态，Hopcroft 分割细化后 BFS 重新编号输出最小完整 DFA。"""
    pos = list(range(len(d.alphabet)))
    order, rows, accept = _reachable_complete(d, d.alphabet, pos)
    m = len(order)
    k = len(d.alphabet)

    # 每个符号的逆迁移：inv[j][q] = { i | rows[i][j] == q }
    inv: list[list[list[int]]] = [
        [[] for _ in range(m)] for _ in range(k)
    ]
    for i in range(m):
        for j in range(k):
            inv[j][rows[i][j]].append(i)

    acc_set = frozenset(i for i in range(m) if accept[i])
    rej_set = frozenset(i for i in range(m) if not accept[i])
    parts: list[frozenset[int]] = [p for p in (acc_set, rej_set) if p]
    part_of: dict[int, frozenset[int]] = {}
    by_id: dict[int, frozenset[int]] = {}
    for p in parts:
        for i in p:
            part_of[i] = p
        by_id[id(p)] = p

    waiting: deque[frozenset[int]] = deque(parts)
    in_wait = {id(p) for p in parts}
    while waiting:
        splitter = waiting.popleft()
        in_wait.discard(id(splitter))
        for j in range(k):
            x_set: set[int] = set()
            for q in splitter:
                x_set.update(inv[j][q])
            if not x_set:
                continue
            touched = {id(part_of[i]) for i in x_set}
            for y_id in touched:
                y_set = by_id[y_id]
                inter = y_set & x_set
                if not inter or inter == y_set:
                    continue
                diff = y_set - inter
                parts.remove(y_set)
                del by_id[y_id]
                parts.append(inter)
                parts.append(diff)
                by_id[id(inter)] = inter
                by_id[id(diff)] = diff
                for i in inter:
                    part_of[i] = inter
                for i in diff:
                    part_of[i] = diff
                if y_id in in_wait:
                    in_wait.discard(y_id)
                    waiting.append(inter)
                    waiting.append(diff)
                    in_wait.add(id(inter))
                    in_wait.add(id(diff))
                else:
                    smaller = inter if len(inter) <= len(diff) else diff
                    waiting.append(smaller)
                    in_wait.add(id(smaller))

    class_of: dict[int, int] = {i: 0 for i in range(m)}
    for ci, p in enumerate(parts):
        for i in p:
            class_of[i] = ci

    # 从初态类出发按 token 顺序 BFS 重新编号，初态为 0。
    start_class = class_of[0]
    new_id = {start_class: 0}
    new_rows: list[list[int]] = []
    new_accept: list[bool] = []
    q: deque[int] = deque([start_class])
    while q:
        c = q.popleft()
        rep = next(iter(parts[c]))
        row: list[int] = []
        for j in range(k):
            # 类中任一代表状态的目标类相同（细化已收敛）。
            dst_class = class_of[rows[rep][j]]
            nc = new_id.get(dst_class)
            if nc is None:
                nc = len(new_id)
                new_id[dst_class] = nc
                q.append(dst_class)
            row.append(nc)
        new_rows.append(row)
        new_accept.append(accept[rep])

    mini = _build_complete(d.alphabet, new_rows, new_accept)

    mapping: dict[int, int] = {}
    sink_state: int | None = None
    for local_i, orig in enumerate(order):
        nid = new_id[class_of[local_i]]
        if orig == SINK:
            sink_state = nid
        else:
            mapping[orig] = nid
    if sink_state is not None:
        mapping[SINK] = sink_state
    return Minimization(mini, mapping, sink_state, sink_state is not None)


# ---------------------------------------------------------------------------
# 执行与判定
# ---------------------------------------------------------------------------


def run(d: DFA, word: Sequence[str]) -> bool:
    """在 DFA 上逐步执行 word；缺失转移（含机器未声明的 token）进入拒绝汇点。

    这与运算时“统一到双方字母表并集，原机未声明 token 进汇点”的语义一致，
    因此可直接用统一字母表上的串在任一原机上执行。
    """
    pos = {t: i for i, t in enumerate(d.alphabet)}
    state = d.start
    for tok in word:
        if not isinstance(tok, str):
            raise DFAError(f"token 必须是字符串: {tok!r}")
        state = d._go(state, pos.get(tok, -1))
    return d._accepts(state)


def _witness_pairs(
    a: DFA,
    b: DFA,
    is_target,
) -> list[str] | None:
    """统一字母表上的产品图 BFS，返回命中 is_target 的最短、字典序最小路径。

    直接在产品状态图上搜索，不按固定长度枚举字符串；同层按 token 字典序
    扩展，故首个命中即最短且字典序最小。无任何命中返回 None（与空列表 ε
    见证区分）。
    """
    sigma, pa, pb = _unify(a, b)
    start = (a.start, b.start)
    if is_target(a, b, *start):
        return []
    parent: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        x, y = q.popleft()
        for j, tok in enumerate(sigma):
            nxt = (a._go(x, pa[j]), b._go(y, pb[j]))
            if nxt in seen:
                continue
            if len(seen) >= MAX_REACHABLE_STATES:
                raise DFALimitError(
                    f"产品图可达状态达到上限 {MAX_REACHABLE_STATES}"
                )
            seen.add(nxt)
            parent[nxt] = ((x, y), tok)
            if is_target(a, b, *nxt):
                # 回溯
                path: list[str] = []
                cur = nxt
                while cur != start:
                    prev, t = parent[cur]
                    path.append(t)
                    cur = prev
                path.reverse()
                return path
            q.append(nxt)
    return None


def witness_inequivalent(a: DFA, b: DFA) -> list[str] | None:
    """L(A)≠L(B) 时返回最短见证（同长度字典序最小）；等价返回 None。

    找到后会在双方机器上实际执行并核对接受性不一致，否则视为内部错误。
    """
    w = _witness_pairs(a, b, lambda ma, mb, x, y: ma._accepts(x) != mb._accepts(y))
    if w is not None:
        ra, rb = run(a, w), run(b, w)
        if ra == rb:
            raise RuntimeError(f"内部校验失败：见证 {w} 在双方接受性相同")
    return w


def witness_not_subset(a: DFA, b: DFA) -> list[str] | None:
    """L(A)⊄L(B) 时返回 A 接受且 B 拒绝的最短见证；包含成立返回 None。"""
    w = _witness_pairs(
        a, b, lambda ma, mb, x, y: ma._accepts(x) and not mb._accepts(y)
    )
    if w is not None:
        ra, rb = run(a, w), run(b, w)
        if not (ra and not rb):
            raise RuntimeError(f"内部校验失败：见证 {w} 不满足 A 接受、B 拒绝")
    return w


def equivalent(a: DFA, b: DFA) -> bool:
    """两机语言是否相同（产品图可达状态中无接受性分歧）。"""
    return witness_inequivalent(a, b) is None


def subset(a: DFA, b: DFA) -> bool:
    """L(A) 是否包含于 L(B)。"""
    return witness_not_subset(a, b) is None
