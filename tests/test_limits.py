"""状态上限：实际探索触发 limit 错误，稀疏可达乘积不被预先拒绝。"""

import unittest

from dfalib import (
    StateLimitError,
    check_equivalent,
    intersect,
    union,
)
from samples import chain, mod_counter


class TestStateLimit(unittest.TestCase):
    def test_limit_error_when_exploration_exceeds(self):
        a = mod_counter(3, 0)
        b = mod_counter(7, 0)
        # 乘积可达 21 个状态，limit=20 在探索到第 21 个时触发
        with self.assertRaises(StateLimitError) as ctx:
            intersect(a, b, limit=20)
        self.assertEqual(ctx.exception.limit, 20)

    def test_exact_limit_succeeds(self):
        a = mod_counter(3, 0)
        b = mod_counter(7, 0)
        d = intersect(a, b, limit=21)  # 恰好 21 个可达状态，允许
        self.assertEqual(len(d.states), 21)
        with self.assertRaises(StateLimitError):
            intersect(a, b, limit=20)

    def test_default_limit_enforced_on_large_product(self):
        # 80、79、77 两两互素，三重乘积可达 80*79*77 > 10000
        big = intersect(mod_counter(80, 0), mod_counter(79, 0))
        self.assertEqual(len(big.states), 80 * 79)
        with self.assertRaises(StateLimitError):
            intersect(big, mod_counter(77, 0))

    def test_sparse_reachable_product_not_pre_rejected(self):
        # 两台 6320 态机器的笛卡尔积约 4 千万 >> 10000，
        # 但同步推进使可达状态只有 6320 个，必须正常完成
        m = intersect(mod_counter(80, 0), mod_counter(79, 0))
        self.assertEqual(len(m.states), 6320)
        p = intersect(m, m)  # 默认 limit=10000，不得按笛卡尔积预先拒绝
        self.assertEqual(len(p.states), 6320)
        self.assertTrue(check_equivalent(p, m).equivalent)

    def test_witness_search_also_respects_limit(self):
        from dfalib import DFA
        a = DFA([0], ["a"], 0, [], [(0, "a", 0)])  # 全拒绝
        b = chain(16)  # 只接受 15 个 "a"：见证搜索必须深入到第 15 层
        with self.assertRaises(StateLimitError):
            check_equivalent(a, b, limit=10)
        # 足够大的 limit 下正常返回最短见证
        r = check_equivalent(a, b, limit=100)
        self.assertFalse(r.equivalent)
        self.assertEqual(r.witness, ["a"] * 15)

    def test_union_limit(self):
        with self.assertRaises(StateLimitError):
            union(mod_counter(4, 0), mod_counter(5, 0), limit=10)


if __name__ == "__main__":
    unittest.main()
