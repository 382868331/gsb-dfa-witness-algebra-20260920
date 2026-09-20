"""测试：输入校验、语言运算（穷举交叉核验）、最小化可区分性、见证与限制。"""

from __future__ import annotations

import itertools
import random
import unittest

import dfa_algebra as da
from dfa_algebra import DFA, DFALimitError, DFAError, SINK


# ---------------------------------------------------------------------------
# 独立参考实现：不依赖产品图，直接在不完整表上逐步执行
# ---------------------------------------------------------------------------


def all_words(alphabet, max_len):
    for n in range(max_len + 1):
        for w in itertools.product(alphabet, repeat=n):
            yield list(w)


def ref_pair_relation(a, b):
    """独立参考：在原始（可能不完整）转移表上枚举可达产品对。

    返回 (reachable, differs, a_only)：是否存在接受性分歧、A 接受 B 拒绝的对。
    与库实现不同，这里直接操作 transitions 字典和 SINK。
    """
    sigma = tuple(sorted(set(a.alphabet) | set(b.alphabet)))

    def go(d, state, tok):
        if state == SINK:
            return SINK
        return d.transitions.get((state, tok), SINK)

    start = (a.start, b.start)
    seen = {start}
    deck = [start]
    i = 0
    differs = (start[0] in a.accepting) != (start[1] in b.accepting)
    a_only = start[0] in a.accepting and start[1] not in b.accepting
    while i < len(deck):
        x, y = deck[i]
        i += 1
        for tok in sigma:
            nx = go(a, x, tok)
            ny = go(b, y, tok)
            pair = (nx, ny)
            if pair in seen:
                continue
            seen.add(pair)
            deck.append(pair)
            ax = nx in a.accepting
            by = ny in b.accepting
            differs = differs or (ax != by)
            a_only = a_only or (ax and not by)
    return differs, a_only


def random_dfa(rng, n_states, alphabet, complete=False, acc_prob=0.4):
    states = list(range(n_states))
    accepting = {s for s in states if rng.random() < acc_prob}
    trans = []
    for s in states:
        for t in alphabet:
            if complete or rng.random() < 0.7:
                trans.append((s, t, rng.randrange(n_states)))
    # 初态恒为 0；保证 0 存在
    return DFA(states, alphabet, 0, accepting, trans)


class ConstructionTests(unittest.TestCase):
    def test_rejects_unknown_transition_target(self):
        with self.assertRaises(DFAError):
            DFA([0, 1], ["a"], 0, [], [(0, "a", 5)])

    def test_rejects_unknown_source_and_accepting(self):
        with self.assertRaises(DFAError):
            DFA([0], ["a"], 0, [], [(1, "a", 0)])
        with self.assertRaises(DFAError):
            DFA([0], ["a"], 0, [1], [])

    def test_rejects_unknown_start(self):
        with self.assertRaises(DFAError):
            DFA([0, 1], ["a"], 3, [], [])

    def test_rejects_duplicate_alphabet_and_transition(self):
        with self.assertRaises(DFAError):
            DFA([0], ["a", "a"], 0, [], [])
        with self.assertRaises(DFAError):
            DFA([0, 1], ["a"], 0, [], [(0, "a", 1), (0, "a", 0)])

    def test_rejects_non_integer_state_and_bool(self):
        with self.assertRaises(DFAError):
            DFA(["0"], ["a"], "0", [], [])
        with self.assertRaises(DFAError):
            DFA([0, 1], ["a"], True, [], [])  # bool 不是合法状态 id

    def test_rejects_bad_token(self):
        with self.assertRaises(DFAError):
            DFA([0, 1], ["a"], 0, [], [(0, "b", 1)])
        with self.assertRaises(DFAError):
            DFA([0], [1], 0, [], [])

    def test_limits(self):
        with self.assertRaises(DFAError):
            DFA(range(81), ["a"], 0, [], [])
        with self.assertRaises(DFAError):
            DFA([0], [f"t{i}" for i in range(9)], 0, [], [])
        # 80 态合法
        DFA(range(80), ["a"], 0, [], [])

    def test_empty_states_rejected(self):
        with self.assertRaises(DFAError):
            DFA([], ["a"], 0, [], [])


class OperationTests(unittest.TestCase):
    def test_missing_transition_goes_to_sink(self):
        d = DFA([0, 1], ["a", "b"], 0, [1], [(0, "a", 1)])
        self.assertTrue(da.run(d, ["a"]))
        self.assertFalse(da.run(d, ["b"]))
        self.assertFalse(da.run(d, ["a", "b"]))  # 态 1 缺 b 转移 -> 汇点
        self.assertFalse(da.run(d, ["a", "a", "a"]))

    def test_different_alphabets_union_lexicographic(self):
        # A: 仅接受 "a"；B: 字母表 {a,b}，仅接受 "a"（b 使其拒绝）
        a = DFA([0, 1, 2], ["a"], 0, [1], [(0, "a", 1), (1, "a", 2), (2, "a", 2)])
        b = DFA([0, 1, 2], ["a", "b"], 0, [1],
                [(0, "a", 1), (0, "b", 2), (1, "a", 2), (1, "b", 2),
                 (2, "a", 2), (2, "b", 2)])
        self.assertTrue(da.equivalent(a, b))
        self.assertEqual(da.intersect(a, b).alphabet, ("a", "b"))
        self.assertIsNone(da.witness_inequivalent(a, b))
        # A 的字母表上补集接受 ε；统一字母表视角下 b 属于双方补集
        ca = da.complement(a)
        self.assertTrue(da.run(ca, []))
        self.assertTrue(da.run(ca, ["a", "a"]))

    def test_operations_vs_bruteforce_random(self):
        rng = random.Random(20260920)
        for seed_case in range(60):
            alpha_a = ["a", "b", "c"][: rng.randrange(0, 4)]
            extra = ["d"] if rng.random() < 0.4 else []
            alpha_b = sorted(set(alpha_a) | set(extra))
            a = random_dfa(rng, rng.randrange(1, 6), alpha_a,
                           complete=rng.random() < 0.5)
            b = random_dfa(rng, rng.randrange(1, 6), alpha_b,
                           complete=rng.random() < 0.5)
            sigma = tuple(sorted(set(alpha_a) | set(alpha_b)))
            ops = {
                "inter": da.intersect(a, b),
                "union": da.union(a, b),
                "diff": da.difference(a, b),
            }
            # 独立参考：产品对闭包给出精确的等价/包含真值
            differs, a_only = ref_pair_relation(a, b)
            for w in all_words(sigma, 5):
                ra, rb = da.run(a, w), da.run(b, w)
                self.assertEqual(da.run(ops["inter"], w), ra and rb, (seed_case, w))
                self.assertEqual(da.run(ops["union"], w), ra or rb, (seed_case, w))
                self.assertEqual(da.run(ops["diff"], w), ra and not rb, (seed_case, w))
            if not differs:
                self.assertIsNone(da.witness_inequivalent(a, b), seed_case)
            else:
                w = da.witness_inequivalent(a, b)
                self.assertIsNotNone(w, seed_case)
                self.assertNotEqual(da.run(a, w), da.run(b, w))
                # 同长度字典序最小：与独立分层枚举结果一致
                ref = next(
                    list(x) for n in range(len(w) + 1)
                    for x in itertools.product(sigma, repeat=n)
                    if da.run(a, list(x)) != da.run(b, list(x))
                )
                self.assertEqual(w, ref, seed_case)
            if not a_only:
                self.assertIsNone(da.witness_not_subset(a, b), seed_case)
            else:
                w = da.witness_not_subset(a, b)
                self.assertTrue(da.run(a, w) and not da.run(b, w))
            # 补集：对 A 显式字母表逐点取反
            comp = da.complement(a)
            for w in all_words(alpha_a, 5):
                self.assertEqual(da.run(comp, w), not da.run(a, w), (seed_case, w))

    def test_all_accept_all_reject(self):
        sigma = ["a", "b"]
        allacc = DFA([0], sigma, 0, [0], [(0, "a", 0), (0, "b", 0)])
        allrej = DFA([0], sigma, 0, [], [(0, "a", 0), (0, "b", 0)])
        self.assertTrue(all(da.run(allacc, w) for w in all_words(sigma, 6)))
        self.assertFalse(any(da.run(allrej, w) for w in all_words(sigma, 6)))
        self.assertTrue(da.equivalent(da.complement(allacc), allrej))
        self.assertTrue(da.equivalent(da.complement(allrej), allacc))
        self.assertTrue(da.equivalent(da.union(allacc, allrej), allacc))
        self.assertTrue(da.equivalent(da.intersect(allacc, allrej), allrej))
        self.assertTrue(da.equivalent(da.difference(allacc, allrej), allacc))


class MinimizationTests(unittest.TestCase):
    def test_removes_unreachable_and_merges_equivalent(self):
        # 0,1 等价（同语言：含 b 即接受）；2 接受自环；3 不可达接受态
        d = DFA([0, 1, 2, 3], ["a", "b"], 0, [2, 3],
                [(0, "a", 1), (0, "b", 2),
                 (1, "a", 1), (1, "b", 2),
                 (2, "a", 2), (2, "b", 2)])
        r = da.minimize(d)
        self.assertEqual(len(r.dfa.states), 2)
        self.assertEqual(r.dfa.start, 0)
        self.assertNotIn(3, r.mapping)  # 不可达态不出现在映射中
        self.assertEqual(r.mapping[0], r.mapping[1])
        self.assertFalse(r.sink_reachable)

    def test_bfs_renumbering_token_order(self):
        # 初态 0：a -> 接受态，b -> 拒绝态；两态自环
        d = DFA([0, 1, 2], ["a", "b"], 0, [2],
                [(0, "a", 2), (0, "b", 1),
                 (1, "a", 1), (1, "b", 1),
                 (2, "a", 2), (2, "b", 2)])
        r = da.minimize(d)
        # BFS：0 -> a 先编号 1（接受），b 编号 2
        self.assertEqual(r.mapping[2], 1)
        self.assertEqual(r.mapping[1], 2)

    def test_sink_mapped_when_reachable(self):
        d = DFA([0], ["x"], 0, [0], [])  # 语言 {ε}
        r = da.minimize(d)
        self.assertEqual(len(r.dfa.states), 2)
        self.assertTrue(r.sink_reachable)
        self.assertIn(SINK, r.mapping)
        self.assertEqual(r.sink_state, r.mapping[SINK])
        self.assertTrue(da.run(r.dfa, []))
        self.assertFalse(da.run(r.dfa, ["x"]))

    def test_sink_merges_with_equivalent_trap(self):
        # 态 1 是显式非接受陷阱，与隐式汇点语义相同
        d = DFA([0, 1], ["x", "y"], 0, [0],
                [(0, "x", 1), (1, "x", 1), (1, "y", 1)])
        r = da.minimize(d)
        self.assertEqual(len(r.dfa.states), 2)
        self.assertTrue(r.sink_reachable)
        self.assertEqual(r.mapping[1], r.mapping[SINK])

    def test_minimized_pairwise_distinguishable_random(self):
        """最小化后任意两状态必须可用某后缀区分（直接在最小机完整表上搜索）。"""
        rng = random.Random(777)
        for _ in range(40):
            alpha = ["a", "b", "c"][: rng.randrange(1, 4)]
            d = random_dfa(rng, rng.randrange(1, 9), alpha,
                           complete=rng.random() < 0.5)
            r = da.minimize(d)
            md = r.dfa
            # 与原机语言一致
            for w in all_words(alpha, 6):
                self.assertEqual(da.run(md, w), da.run(d, w))
            n = len(md.states)
            acc = [md._acc[i] for i in range(n)]
            rows = md._rows
            # 最小性：再次最小化状态数不变
            self.assertEqual(len(da.minimize(md).dfa.states), n)
            # 每对状态存在区分后缀（对状态对做 BFS）
            for p in range(n):
                for q in range(p + 1, n):
                    suffix = self._distinguishing_suffix(rows, acc, p, q, len(alpha))
                    self.assertIsNotNone(suffix, (p, q, alpha))

    @staticmethod
    def _distinguishing_suffix(rows, acc, p, q, k):
        if acc[p] != acc[q]:
            return []
        seen = {(p, q)}
        deck = [(p, q, [])]
        i = 0
        while i < len(deck):
            x, y, w = deck[i]
            i += 1
            for sym in range(k):
                nx, ny = rows[x][sym], rows[y][sym]
                if acc[nx] != acc[ny]:
                    return w + [sym]
                if (nx, ny) not in seen and nx != ny:
                    seen.add((nx, ny))
                    deck.append((nx, ny, w + [sym]))
        return None

    def test_minimize_supports_large_results(self):
        # 运算结果（远超 80 态）也必须能最小化与查询
        d = self._grid_dfa(40)  # 40*40 = 1600 态
        r = da.minimize(d)
        self.assertEqual(len(r.dfa.states), 1)  # 网格语言等价于单态全接受
        self.assertTrue(da.run(r.dfa, ["a", "b", "a"]))

    @staticmethod
    def _grid_dfa(size):
        pa = DFA(range(size), ["a", "b"], 0, list(range(size)),
                 [(i, "a", min(i + 1, size - 1)) for i in range(size)]
                 + [(i, "b", i) for i in range(size)])
        pb = DFA(range(size), ["a", "b"], 0, list(range(size)),
                 [(i, "b", min(i + 1, size - 1)) for i in range(size)]
                 + [(i, "a", i) for i in range(size)])
        return da.intersect(pa, pb)


class WitnessTests(unittest.TestCase):
    def test_epsilon_witness_is_empty_list(self):
        a = DFA([0], ["a"], 0, [0], [(0, "a", 0)])   # 接受 ε（Σ*）
        b = DFA([0], ["a"], 0, [], [(0, "a", 0)])    # 拒绝 ε（∅）
        self.assertEqual(da.witness_inequivalent(a, b), [])
        self.assertFalse(da.equivalent(a, b))
        # b⊆a 成立无反例；a⊄b 的最短反例就是 ε（空列表）
        self.assertIsNone(da.witness_not_subset(b, a))
        self.assertEqual(da.witness_not_subset(a, b), [])
        # None 与 [] 不混淆
        self.assertIsNone(da.witness_inequivalent(a, a))

    def test_shortest_lexicographic_witness(self):
        # A 拒绝一切；B 接受所有以 c 开头或以 bb 开头的串……构造已知见证
        # B：接受恰为 ["c"]、["b", "b"]、["b", "a"] 等；最短分歧长度 1，
        # 字母序 a < b < c，令 B 接受 "c" 且不接受 "a"/"b"。
        a = DFA([0], ["a", "b", "c"], 0, [],
                [(0, t, 0) for t in ["a", "b", "c"]])
        b = DFA([0, 1, 2], ["a", "b", "c"], 0, [1],
                [(0, "c", 1), (0, "a", 2), (0, "b", 2)]
                + [(1, t, 2) for t in ["a", "b", "c"]]
                + [(2, t, 2) for t in ["a", "b", "c"]])
        self.assertEqual(da.witness_inequivalent(a, b), ["c"])

    def test_lexicographic_tiebreak_at_same_length(self):
        # 最短见证长度 2，候选 "ba" 与 "bb"；"ba" 字典序更小
        # B：态 0 上 a->陷阱2，b->态3；态3 上 a->接受1，b->陷阱2，c->陷阱2
        a = DFA([0], ["a", "b", "c"], 0, [],
                [(0, t, 0) for t in ["a", "b", "c"]])
        b = DFA([0, 1, 2, 3], ["a", "b", "c"], 0, [1],
                [(0, "a", 2), (0, "b", 3), (0, "c", 2),
                 (3, "a", 1), (3, "b", 2), (3, "c", 2)]
                + [(1, t, 2) for t in ["a", "b", "c"]]
                + [(2, t, 2) for t in ["a", "b", "c"]])
        w = da.witness_inequivalent(a, b)
        self.assertEqual(w, ["b", "a"])
        # 与独立穷举一致
        sigma = ["a", "b", "c"]
        ref = next(list(x) for n in range(6)
                   for x in itertools.product(sigma, repeat=n)
                   if da.run(a, list(x)) != da.run(b, list(x)))
        self.assertEqual(w, ref)
        # 见证确实在双方上接受性不同
        self.assertNotEqual(da.run(a, w), da.run(b, w))

    def test_subset_witness_direction(self):
        # A = {"a"}, B = {"a","aa"}：A⊆B 成立；B⊆A 不成立，见证 aa
        a = DFA([0, 1, 2], ["a"], 0, [1],
                [(0, "a", 1), (1, "a", 2), (2, "a", 2)])
        b = DFA([0, 1, 2], ["a"], 0, [1],
                [(0, "a", 1), (1, "a", 1)])
        self.assertTrue(da.subset(a, b))
        self.assertFalse(da.subset(b, a))
        self.assertEqual(da.witness_not_subset(b, a), ["a", "a"])
        self.assertIsNone(da.witness_not_subset(a, b))


class EmptyAlphabetTests(unittest.TestCase):
    def test_empty_alphabet_languages(self):
        eps = DFA([0], [], 0, [0], [])   # {ε}
        empty = DFA([0], [], 0, [], [])  # ∅
        self.assertTrue(da.run(eps, []))
        self.assertFalse(da.run(empty, []))
        self.assertFalse(da.equivalent(eps, empty))
        self.assertEqual(da.witness_inequivalent(eps, empty), [])
        self.assertTrue(da.equivalent(da.complement(eps), empty))
        self.assertTrue(da.equivalent(da.complement(empty), eps))
        # {ε} 与“有字母但永远拒绝”的机器语言相同
        rej = DFA([0], ["z"], 0, [], [(0, "z", 0)])
        self.assertTrue(da.equivalent(empty, rej))
        # {ε} 与“初态接受但读 z 进陷阱”的机器语言相同
        eps_z = DFA([0, 1], ["z"], 0, [0], [(0, "z", 1), (1, "z", 1)])
        self.assertTrue(da.equivalent(eps, eps_z))

    def test_minimize_empty_alphabet(self):
        r = da.minimize(DFA([0, 1], [], 0, [0], []))
        self.assertEqual(len(r.dfa.states), 1)
        self.assertTrue(da.run(r.dfa, []))


class LimitTests(unittest.TestCase):
    @staticmethod
    def _paths(size_a, size_b):
        """两台各 80 态以内的网格机，乘积可达 size_a*size_b 个不同对。

        第三个 token c 在两台机器上都自环：同一位置可用任意多 c 步到达，
        因此再与“每步翻转”的奇偶机求交时每个网格位置有两种奇偶。
        """
        pa = DFA(range(size_a), ["a", "b", "c"], 0, list(range(size_a)),
                 [(i, "a", min(i + 1, size_a - 1)) for i in range(size_a)]
                 + [(i, "b", i) for i in range(size_a)]
                 + [(i, "c", i) for i in range(size_a)])
        pb = DFA(range(size_b), ["a", "b", "c"], 0, list(range(size_b)),
                 [(i, "b", min(i + 1, size_b - 1)) for i in range(size_b)]
                 + [(i, "a", i) for i in range(size_b)]
                 + [(i, "c", i) for i in range(size_b)])
        return da.intersect(pa, pb)

    def test_sparse_product_above_cartesian_not_rejected(self):
        # D1 有 6400 个可达态；与 2 态机的笛卡尔积 12800 > 10000，
        # 但 z 立即把 D1 打入汇点，实际只探索 6401 个可达对——不得报错。
        d1 = self._paths(80, 80)
        self.assertEqual(len(d1.states), 6400)
        d2 = DFA([0, 1], ["z"], 0, [], [(0, "z", 1), (1, "z", 1)])
        result = da.intersect(d1, d2)
        self.assertLessEqual(len(result.states), da.MAX_REACHABLE_STATES)

    def test_limit_raised_only_when_new_state_needed(self):
        # 6400 个网格位置各有两种可达奇偶：12800 个可达对，
        # 探索到第 10001 个新状态时才报 limit（不预先按笛卡尔积拒绝）。
        d1 = self._paths(80, 80)
        parity = DFA([0, 1], ["a", "b", "c"], 0, [],
                     [(0, t, 1) for t in ["a", "b", "c"]]
                     + [(1, t, 0) for t in ["a", "b", "c"]])
        with self.assertRaises(DFALimitError):
            da.intersect(d1, parity)
        # 见证搜索同样在产品图上探索：用两态都接受的同构奇偶机，
        # 接受性永不分歧，等价搜索必须走遍全部 12800 对后才触限。
        parity_acc = DFA([0, 1], ["a", "b", "c"], 0, [0, 1],
                         [(0, t, 1) for t in ["a", "b", "c"]]
                         + [(1, t, 0) for t in ["a", "b", "c"]])
        with self.assertRaises(DFALimitError):
            da.witness_inequivalent(d1, parity_acc)

    def test_small_product_below_limit_ok(self):
        d = self._paths(40, 40)  # 1600 态
        self.assertEqual(len(d.states), 1600)
        r = da.minimize(d)
        self.assertGreaterEqual(len(r.dfa.states), 1)


if __name__ == "__main__":
    unittest.main()
