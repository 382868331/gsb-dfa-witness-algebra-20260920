"""DFA 语言代数演示：一个等价（正常结果）与一个实际触发的不等价失败。

运行：python demo.py
所有结论均由 dfa_algebra 实际计算，并用穷举短串独立复核。
"""

from __future__ import annotations

import itertools
import time

import dfa_algebra as da


def enumerate_words(alphabet, max_len):
    for length in range(max_len + 1):
        for word in itertools.product(alphabet, repeat=length):
            yield list(word)


def shortest_witness_bruteforce(a, b, max_len=8):
    """独立参考实现：按长度分层、同层字典序枚举，找接受性分歧的最短串。"""
    sigma = tuple(sorted(set(a.alphabet) | set(b.alphabet)))
    for length in range(max_len + 1):
        for word in itertools.product(sigma, repeat=length):
            w = list(word)
            if da.run(a, w) != da.run(b, w):
                return w
    return None


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 规则 A（不完整转移表）：字母表 {a,b}，语言 = 含 a 且不含 c。
    # 未声明的 c 一律落入隐式拒绝汇点；状态 1 对 a,b 自环。
    # ------------------------------------------------------------------
    a = da.DFA(
        states=[0, 1],
        alphabet=["a", "b"],
        start=0,
        accepting=[1],
        transitions=[
            (0, "a", 1),
            (0, "b", 0),
            (1, "a", 1),
            (1, "b", 1),
        ],
    )

    # ------------------------------------------------------------------
    # 规则 B（另一来源，结构冗余、含显式陷阱态和不可达态）：字母表 {a,b,c}。
    # 3 是显式非接受陷阱（c 一旦出现即永久拒绝，与 A 的隐式汇点同语义）；
    # 4 不可达，最小化时必须被删除。
    # ------------------------------------------------------------------
    b = da.DFA(
        states=[0, 1, 2, 3, 4],
        alphabet=["a", "b", "c"],
        start=0,
        accepting=[2, 4],
        transitions=[
            (0, "a", 2), (0, "b", 1), (0, "c", 3),
            (1, "a", 2), (1, "b", 1), (1, "c", 3),
            (2, "a", 2), (2, "b", 2), (2, "c", 3),
            (3, "a", 3), (3, "b", 3), (3, "c", 3),
        ],
    )

    # ------------------------------------------------------------------
    # 规则 C：语言 = {b}（只接受恰好一个 b）。
    # ------------------------------------------------------------------
    c = da.DFA(
        states=[0, 1, 2],
        alphabet=["a", "b", "c"],
        start=0,
        accepting=[1],
        transitions=[
            (0, "a", 2), (0, "b", 1), (0, "c", 2),
            (1, "a", 2), (1, "b", 2), (1, "c", 2),
            (2, "a", 2), (2, "b", 2), (2, "c", 2),
        ],
    )

    print("=" * 68)
    print("场景一（正常结果）：独立产生的规则 A 与 B 应描述同一语言")
    print("=" * 68)
    print(f"A: 字母表 {list(a.alphabet)}，{len(a.states)} 态，转移表不完整")
    print(f"B: 字母表 {list(b.alphabet)}，{len(b.states)} 态（含不可达态 4）")

    ma = da.minimize(a)
    mb = da.minimize(b)
    print(f"最小化后：A {len(ma.dfa.states)} 态，B {len(mb.dfa.states)} 态")
    print(f"A 对自身字母表 {list(a.alphabet)} 转移完整：隐式汇点可达={ma.sink_reachable}，映射 {dict(sorted(ma.mapping.items()))}")
    print(f"B 含显式陷阱态 3、不可达接受态 4：汇点可达={mb.sink_reachable}，映射 {dict(sorted(mb.mapping.items()))}（态 4 已删除，0/1 合并）")

    # 真正不完整的小机 D：字母表 {x}，初态接受且没有任何转移。
    # 语言 = {ε}；x 使隐式汇点可达，最小化后映射中必须标明 SINK。
    d = da.DFA(states=[0], alphabet=["x"], start=0, accepting=[0], transitions=[])
    md = da.minimize(d)
    print(f"D（无转移，语言={{ε}}）最小化：{len(md.dfa.states)} 态，汇点可达={md.sink_reachable}，"
          f"映射 {dict(sorted(md.mapping.items()))}（键 {da.SINK} 即隐式汇点）")

    w = da.witness_inequivalent(a, b)
    print(f"产品图最短见证搜索：{w!r}（None 表示无见证，即等价）")

    # 独立穷举复核（统一字母表 {a,b,c}，长度 <= 6，共 1093 条串）
    sigma = sorted(set(a.alphabet) | set(b.alphabet))
    mismatch = [w2 for w2 in enumerate_words(sigma, 6) if da.run(a, w2) != da.run(b, w2)]
    total = sum(len(sigma) ** i for i in range(7))
    print(f"独立穷举复核：{total} 条长度≤6 的串，接受性分歧 {len(mismatch)} 条 -> "
          f"{'一致' if not mismatch and da.equivalent(a, b) else '不一致!'}")

    # 顺手展示运算：交集/并集/差集/补集都可用，且差集为空语言
    diff_ab = da.difference(a, b)
    diff_ba = da.difference(b, a)
    empty_words = [w2 for w2 in enumerate_words(sigma, 6) if da.run(diff_ab, w2) or da.run(diff_ba, w2)]
    print(f"L(A)\\L(B) 与 L(B)\\L(A) 短串接受数：{len(empty_words)}（两个方向均为空语言）")
    comp = da.complement(a)
    print(f"补集抽查（A 的字母表 {{a,b}} 上）：complement(A) 接受 ε={da.run(comp, [])}、'b'={da.run(comp, ['b'])}、'a'={da.run(comp, ['a'])}")

    print()
    print("=" * 68)
    print("场景二（实际触发的失败）：规则 A 与 C 不等价")
    print("=" * 68)
    print(f"C: 字母表 {list(c.alphabet)}，只接受恰好 ['b']")

    witness = da.witness_inequivalent(a, c)
    print(f"产品图 BFS 返回的最短见证：{witness}")
    assert witness is not None, "应当找到见证"

    # 在双方机器上实际执行并核对接受性
    ra, rc = da.run(a, witness), da.run(c, witness)
    print(f"复核接受性：A({witness}) = {ra}，C({witness}) = {rc}，确实分歧：{ra != rc}")

    # 包含关系同样失败，给出各自方向的见证
    w_ac = da.witness_not_subset(a, c)
    w_ca = da.witness_not_subset(c, a)
    print(f"L(C) 包含于 L(A)？反例 L(A)\\L(C) 最短见证: {w_ac}（A 接受而 C 拒绝）")
    print(f"L(A) 包含于 L(C)？反例 L(C)\\L(A) 最短见证: {w_ca}（C 接受而 A 拒绝）")

    # 独立分层穷举复核“最短且同长度字典序最小”
    ref = shortest_witness_bruteforce(a, c)
    print(f"独立穷举参考实现得到的最短/字典序最小见证：{ref}")
    print(f"与产品图结果一致：{witness == ref}")

    print()
    print(f"演示完成，用时 {time.perf_counter() - t0:.2f} 秒。")


if __name__ == "__main__":
    main()
