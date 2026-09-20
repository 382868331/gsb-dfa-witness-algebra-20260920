"""等价/包含判断与最短见证的测试。"""

import unittest

from dfalib import DFA, check_equivalent, check_included
from samples import ends_with
from tests.helpers import all_strings


def brute_force_first_difference(a, b, max_len):
    """按 (长度, 字典序) 枚举，找第一个接受性不同的串；独立参考实现。"""
    alphabet = sorted(set(a.alphabet) | set(b.alphabet))
    for w in all_strings(alphabet, max_len):
        if a.accepts(w) != b.accepts(w):
            return w
    return None


class TestWitness(unittest.TestCase):
    def test_equivalent_machines_have_no_witness(self):
        a = ends_with("abb", ["a", "b"])
        b = ends_with("abb", ["a", "b"])
        r = check_equivalent(a, b)
        self.assertTrue(r.equivalent)
        self.assertIsNone(r.witness)  # None 表示不存在见证

    def test_inequivalent_machines_shortest_witness(self):
        a = ends_with("abb", ["a", "b"])
        b = ends_with("ab", ["a", "b"])
        r = check_equivalent(a, b)
        self.assertFalse(r.equivalent)
        self.assertEqual(r.witness, ["a", "b"])  # b 接受 "ab"，a 不接受
        self.assertFalse(r.accepted_by_a)
        self.assertTrue(r.accepted_by_b)
        # 与独立枚举参考一致
        self.assertEqual(r.witness, brute_force_first_difference(a, b, 6))

    def test_empty_list_is_valid_witness(self):
        a = DFA([0], ["a"], 0, [0], [(0, "a", 0)])  # 接受空串
        b = DFA([0], ["a"], 0, [], [(0, "a", 0)])   # 全拒绝
        r = check_equivalent(a, b)
        self.assertFalse(r.equivalent)
        self.assertIsNotNone(r.witness)
        self.assertEqual(r.witness, [])  # 空列表是合法见证，不是 None
        self.assertTrue(r.accepted_by_a)
        self.assertFalse(r.accepted_by_b)

    def test_lexicographically_smallest_among_shortest(self):
        # 双方只在长度 2 的串上不同，同长度应取字典序最小的 "ab"
        a = DFA([0, 1, 2, 3], ["a", "b"], 0, [2],
                [(0, "a", 1), (1, "b", 2), (0, "b", 3), (3, "a", 2)])
        b = DFA([0], ["a", "b"], 0, [], [(0, "a", 0), (0, "b", 0)])
        r = check_equivalent(a, b)
        self.assertEqual(r.witness, ["a", "b"])
        self.assertEqual(r.witness, brute_force_first_difference(a, b, 4))

    def test_witness_matches_brute_force_on_random_machines(self):
        import random
        from tests.helpers import random_dfa
        for seed in (3, 5, 8, 13, 21):
            rng = random.Random(seed)
            a = random_dfa(rng, ["a", "b"], 4)
            b = random_dfa(rng, ["a", "b"], 4)
            with self.subTest(seed=seed):
                r = check_equivalent(a, b)
                ref = brute_force_first_difference(a, b, 8)
                if ref is None:
                    self.assertTrue(r.equivalent)
                else:
                    self.assertFalse(r.equivalent)
                    self.assertEqual(r.witness, ref)
                    self.assertEqual(r.accepted_by_a, a.accepts(ref))
                    self.assertEqual(r.accepted_by_b, b.accepts(ref))

    def test_inclusion_witness(self):
        a = ends_with("ab", ["a", "b"])
        b = ends_with("b", ["a", "b"])
        # L(a) ⊆ L(b)：以 "ab" 结尾必然以 "b" 结尾
        self.assertTrue(check_included(a, b).contained)
        # 反向不成立：见证是 L(b) \ L(a) 中最短的 "b"
        r = check_included(b, a)
        self.assertFalse(r.contained)
        self.assertEqual(r.witness, ["b"])
        self.assertTrue(r.accepted_by_a)   # 属于 L(b)
        self.assertFalse(r.accepted_by_b)  # 不属于 L(a)

    def test_inclusion_empty_string_counterexample(self):
        a = DFA([0], ["a"], 0, [0], [(0, "a", 0)])
        b = DFA([0], ["a"], 0, [], [(0, "a", 0)])
        r = check_included(a, b)
        self.assertFalse(r.contained)
        self.assertEqual(r.witness, [])


if __name__ == "__main__":
    unittest.main()
