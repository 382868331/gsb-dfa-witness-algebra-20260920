"""最小化的正确性测试：可达性、汇点标注、BFS 编号、可区分性。"""

import itertools
import random
import unittest

from dfalib import DFA, are_equivalent, minimize, union
from samples import ends_with
from tests.helpers import all_strings, distinguishing_suffix, random_dfa


class TestMinimize(unittest.TestCase):
    def test_redundant_and_unreachable_states_removed(self):
        # 与 ends_with("abb") 等价，但多一个可达冗余态和一个不可达态
        base = ends_with("abb", ["a", "b"])
        triples = [(s, t, d) for s, row in base.transitions.items()
                   for t, d in row.items()
                   if not (s == 0 and t == "b")]
        # 0 的 "b" 改指向新态 4；4 复制 0 的行为（与 0 等价、可合并）
        triples += [(0, "b", 4), (4, "a", 1), (4, "b", 4)]
        triples += [(5, "a", 5), (5, "b", 5)]  # 不可达
        redundant = DFA([0, 1, 2, 3, 4, 5], ["a", "b"], 0, [3], triples)
        m = minimize(redundant)
        self.assertEqual(len(m.dfa.states), 4)  # "abb" 后缀自动机的最小规模
        self.assertNotIn(5, m.state_map)        # 不可达状态不进映射
        self.assertEqual(m.state_map[0], m.state_map[4])  # 冗余态被合并
        self.assertTrue(are_equivalent(m.dfa, base))

    def test_start_renumbered_to_zero_bfs_order(self):
        # 初态不是 0 的机器，最小化后初态必须是 0
        d = DFA([0, 1], ["a"], 1, [1], [(1, "a", 1)])
        m = minimize(d)
        self.assertEqual(m.dfa.start, 0)
        self.assertEqual(len(m.dfa.states), 1)
        self.assertEqual(m.state_map, {1: 0})

    def test_implicit_sink_marked_when_reachable(self):
        # 缺转移 => 隐式汇点可达，必须标明
        d = DFA([0, 1], ["a", "b"], 0, [1], [(0, "a", 1)])
        m = minimize(d)
        self.assertIsNotNone(m.sink_state)
        sink = m.sink_state
        self.assertNotIn(sink, m.dfa.accepting)
        # 汇点在最小化机中是自环
        for tok in m.dfa.alphabet:
            self.assertEqual(m.dfa.transitions[sink][tok], sink)
        # 最小化结果是完备机
        for s in m.dfa.states:
            for tok in m.dfa.alphabet:
                self.assertIn(tok, m.dfa.transitions[s])

    def test_no_sink_when_already_complete(self):
        m = minimize(ends_with("ab", ["a", "b"]))
        self.assertIsNone(m.sink_state)

    def test_all_reject_minimizes_to_single_sink_like_state(self):
        d = DFA([0, 1, 2], ["a"], 0, [], [(0, "a", 1), (1, "a", 2), (2, "a", 2)])
        m = minimize(d)
        self.assertEqual(len(m.dfa.states), 1)
        self.assertEqual(m.dfa.accepting, frozenset())

    def test_minimized_states_pairwise_distinguishable(self):
        # 随机小机取并后最小化，任意两状态都必须有可区分后缀
        for seed in (5, 6, 7, 8):
            rng = random.Random(seed)
            a = random_dfa(rng, ["a", "b"], 5)
            b = random_dfa(rng, ["a", "b"], 5)
            m = minimize(union(a, b))
            with self.subTest(seed=seed):
                for p, q in itertools.combinations(sorted(m.dfa.states), 2):
                    suffix = distinguishing_suffix(m.dfa, p, q)
                    self.assertIsNotNone(suffix, (p, q))
                    # 后缀确实把两者分开
                    def run(s):
                        cur = s
                        for tok in suffix:
                            cur = m.dfa.transitions[cur][tok]
                        return cur in m.dfa.accepting
                    self.assertNotEqual(run(p), run(q))

    def test_state_map_consistent_with_acceptance(self):
        # 原可达状态与其映像对任意短串的后续接受性一致
        rng = random.Random(11)
        a = random_dfa(rng, ["a", "b"], 6)
        m = minimize(a)
        for orig, new in m.state_map.items():
            for w in all_strings(list(a.alphabet), 3):
                cur_o, cur_n = orig, new
                for tok in w:
                    cur_o = a.transitions.get(cur_o, {}).get(tok)
                    cur_n = m.dfa.transitions[cur_n][tok]
                o_acc = cur_o is not None and cur_o in a.accepting
                n_acc = cur_n in m.dfa.accepting
                self.assertEqual(o_acc, n_acc, (orig, w))


if __name__ == "__main__":
    unittest.main()
