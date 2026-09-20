"""演示：一个正常结果 + 两个实际触发的失败（不等价见证、状态上限）。

运行：python demo.py
所有输出均由 dfalib 实际计算。
"""

import sys

import dfalib as d
from samples import ends_with, mod_counter


def main():
    # Windows 控制台默认 GBK，统一切到 UTF-8 以正常输出中文与符号
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 64)
    print("1. 正常结果：两份独立产生的规则被判定等价，并给出最小化")
    print("=" * 64)
    # 规则一：标准的 "以 abb 结尾" 自动机（4 态，完备）
    m1 = ends_with("abb", ["a", "b"])
    # 规则二：同一语言的另一种写法，含可达冗余态 4 与不可达态 5
    base = [(s, t, dst) for s, row in m1.transitions.items()
            for t, dst in row.items() if not (s == 0 and t == "b")]
    triples = base + [(0, "b", 4), (4, "a", 1), (4, "b", 4),
                      (5, "a", 5), (5, "b", 5)]
    m2 = d.DFA([0, 1, 2, 3, 4, 5], ["a", "b"], 0, [3], triples)
    print(f"机器一: {m1!r}")
    print(f"机器二: {m2!r}  (含冗余态 4、不可达态 5)")

    eq = d.check_equivalent(m1, m2)
    print(f"check_equivalent -> equivalent={eq.equivalent}, witness={eq.witness}")

    m = d.minimize(m2)
    print(f"minimize(机器二): {len(m2.states)} 态 -> {len(m.dfa.states)} 态, "
          f"初态={m.dfa.start}, 隐式汇点={m.sink_state}")
    print(f"  原可达状态 -> 最小化状态映射: {dict(sorted(m.state_map.items()))}")

    inter = d.intersect(m1, m2)
    uni = d.union(m1, m2)
    diff = d.difference(m1, m2)
    print(f"交/并/差可达状态数: {len(inter.states)}/{len(uni.states)}/"
          f"{len(diff.states)}")
    inc = d.check_included(m1, m2)
    print(f"L(机器一) ⊆ L(机器二): {inc.contained}")
    # 直接验证差集是空语言：与全拒绝机等价
    empty = d.DFA([0], ["a", "b"], 0, [], [(0, "a", 0), (0, "b", 0)])
    print(f"差集等价于空语言: {d.are_equivalent(diff, empty)}")

    # 缺转移机器的最小化：隐式汇点可达，必须标明
    partial = d.DFA([0, 1], ["a", "b"], 0, [1], [(0, "a", 1)])
    mp = d.minimize(partial)
    print(f"缺转移机器最小化: {len(mp.dfa.states)} 态, "
          f"映射={dict(sorted(mp.state_map.items()))}, "
          f"隐式汇点 -> 状态 {mp.sink_state}")

    print()
    print("=" * 64)
    print("2. 实际触发的失败一：两台机器不等价，返回最短见证")
    print("=" * 64)
    m3 = ends_with("ab", ["a", "b"])
    print(f"机器三: {m3!r}  (以 ab 结尾)")
    r = d.check_equivalent(m1, m3)
    print(f"check_equivalent(机器一, 机器三) -> equivalent={r.equivalent}")
    print(f"  最短见证: {r.witness}  (空列表也是合法见证, 与 None 不同)")
    print(f"  机器一接受该见证: {r.accepted_by_a}, "
          f"机器三接受该见证: {r.accepted_by_b}")
    # 复核：把见证逐 token 在双方上重跑
    print(f"  复核 accepts: 机器一={m1.accepts(r.witness)}, "
          f"机器三={m3.accepts(r.witness)}")
    r2 = d.check_included(m3, m1)
    print(f"L(机器三) ⊆ L(机器一): {r2.contained}, 见证={r2.witness}")

    print()
    print("=" * 64)
    print("3. 实际触发的失败二：单次运算状态上限")
    print("=" * 64)
    big = d.intersect(mod_counter(80, 0), mod_counter(79, 0))
    print(f"mod80 ∩ mod79 -> {len(big.states)} 个可达状态 (默认上限 "
          f"{d.DEFAULT_STATE_LIMIT})")
    try:
        d.intersect(big, mod_counter(77, 0))
        print("未触发上限（不应出现）")
    except d.StateLimitError as e:
        print(f"再交 mod77 -> 实际触发 StateLimitError: {e}")
    # 稀疏可达乘积不按笛卡尔积总大小预先拒绝
    p = d.intersect(big, big)
    n = len(big.states)
    print(f"稀疏乘积: 双方各 {n} 态, 笛卡尔积 {n * n} 远超上限, "
          f"实际可达 {len(p.states)} 态, 正常完成")


if __name__ == "__main__":
    main()
