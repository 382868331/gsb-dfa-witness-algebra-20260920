"""交、并、差、补的语义测试与小机穷举交叉核验。"""

import random
import unittest

from dfalib import (
    DFA,
    are_equivalent,
    check_included,
    complement,
    difference,
    intersect,
    minimize,
    union,
)
from tests.helpers import all_strings, random_dfa


def all_accept(alphabet):
    return DFA([0], list(alphabet), 0, [0], [(0, t, 0) for t in alphabet])


def all_reject(alphabet):
    return DFA([0], list(alphabet), 0, [], [(0, t, 0) for t in alphabet])


class TestBasicOps(unittest.TestCase):
    def test_disjoint_alphabets_union_and_intersection(self):
        a = DFA([0, 1], ["a"], 0, [1], [(0, "a", 1), (1, "a", 1)])
        b = DFA([0, 1], ["b"], 0, [1], [(0, "b", 1), (1, "b", 1)])
        u = union(a, b)
        self.assertEqual(u.alphabet, ("a", "b"))
        self.assertTrue(u.accepts(["a"]))
        self.assertTrue(u.accepts(["b"]))
        self.assertFalse(u.accepts(["a", "b"]))
        i = intersect(a, b)
        # 双方字母表不同，交集为空语言
        self.assertTrue(are_equivalent(i, all_reject(["a", "b"])))

    def test_all_accept_and_all_reject(self):
        aa, ar = all_accept(["a", "b"]), all_reject(["a", "b"])
        self.assertTrue(are_equivalent(complement(aa), ar))
        self.assertTrue(are_equivalent(complement(ar), aa))
        self.assertTrue(are_equivalent(union(aa, ar), aa))
        self.assertTrue(are_equivalent(intersect(aa, ar), ar))
        self.assertTrue(are_equivalent(difference(aa, ar), aa))
        self.assertTrue(are_equivalent(difference(ar, aa), ar))

    def test_complement_over_empty_alphabet(self):
        # 空字母表下全部有限串只有空串
        accepts_empty = DFA([0], [], 0, [0], [])
        rejects_empty = DFA([0], [], 0, [], [])
        self.assertTrue(are_equivalent(complement(accepts_empty), rejects_empty))
        self.assertTrue(are_equivalent(complement(rejects_empty), accepts_empty))

    def test_difference_witness_direction(self):
        # L(a) = 含 "a"，L(b) = 全接受 => 差为空
        a = DFA([0, 1], ["a"], 0, [1], [(0, "a", 1), (1, "a", 1)])
        self.assertTrue(check_included(a, all_accept(["a"])).contained)
        self.assertFalse(check_included(all_accept(["a"]), a).contained)


class TestCrossCheck(unittest.TestCase):
    """固定种子随机小机：穷举短串，把运算结果与逐串语义组合对比。"""

    SEEDS = (1, 2, 3, 17, 42, 99)

    def test_ops_match_pointwise_semantics(self):
        for seed in self.SEEDS:
            rng = random.Random(seed)
            a = random_dfa(rng, ["a", "b"], 5)
            b = random_dfa(rng, ["b", "c"], 5)  # 字母表不同
            with self.subTest(seed=seed):
                inter = intersect(a, b)
                uni = union(a, b)
                diff = difference(a, b)
                comp = complement(a)
                union_alpha = sorted(set(a.alphabet) | set(b.alphabet))
                for w in all_strings(union_alpha, 4):
                    x, y = a.accepts(w), b.accepts(w)
                    self.assertEqual(inter.accepts(w), x and y, w)
                    self.assertEqual(uni.accepts(w), x or y, w)
                    self.assertEqual(diff.accepts(w), x and not y, w)
                for w in all_strings(list(a.alphabet), 4):
                    self.assertEqual(comp.accepts(w), not a.accepts(w), w)

    def test_minimize_preserves_language(self):
        for seed in self.SEEDS:
            rng = random.Random(seed + 1000)
            a = random_dfa(rng, ["a", "b"], 6)
            b = random_dfa(rng, ["b"], 4)
            uni = union(a, b)
            m = minimize(uni)
            with self.subTest(seed=seed):
                for w in all_strings(list(uni.alphabet), 4):
                    self.assertEqual(m.dfa.accepts(w), uni.accepts(w), w)
                # 最小化结果与原运算结果等价（乘积图判断，非枚举）
                self.assertTrue(are_equivalent(m.dfa, uni))

    def test_de_morgan_on_random_machines(self):
        for seed in self.SEEDS:
            rng = random.Random(seed + 2000)
            a = random_dfa(rng, ["a", "b"], 4)
            b = random_dfa(rng, ["a", "b"], 4)
            with self.subTest(seed=seed):
                # ~(A ∪ B) == ~A ∩ ~B
                lhs = complement(union(a, b))
                rhs = intersect(complement(a), complement(b))
                self.assertTrue(are_equivalent(lhs, rhs))
                # A \ B == A ∩ ~B
                self.assertTrue(
                    are_equivalent(difference(a, b), intersect(a, complement(b)))
                )


if __name__ == "__main__":
    unittest.main()
