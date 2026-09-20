"""DFA 构造与校验的边界测试。"""

import unittest

from dfalib import (
    DFA,
    DEFAULT_IMPORT_MAX_STATES,
    MAX_ALPHABET,
    ValidationError,
)


class TestValidation(unittest.TestCase):
    def test_non_integer_state_id_rejected(self):
        with self.assertRaises(ValidationError):
            DFA(["q0"], ["a"], "q0", [], [])
        with self.assertRaises(ValidationError):
            DFA([True], ["a"], True, [], [])  # bool 不算整数状态 id

    def test_duplicate_alphabet_token_rejected(self):
        with self.assertRaises(ValidationError):
            DFA([0], ["a", "a"], 0, [], [])

    def test_alphabet_size_limit(self):
        with self.assertRaises(ValidationError):
            DFA([0], [str(i) for i in range(MAX_ALPHABET + 1)], 0, [], [])
        # 恰好 8 个合法
        DFA([0], [str(i) for i in range(MAX_ALPHABET)], 0, [], [])

    def test_non_string_token_rejected(self):
        with self.assertRaises(ValidationError):
            DFA([0], [1], 0, [], [])

    def test_unknown_state_references_rejected(self):
        with self.assertRaises(ValidationError):
            DFA([0], ["a"], 1, [], [])  # 初态未知
        with self.assertRaises(ValidationError):
            DFA([0], ["a"], 0, [9], [])  # 接受态未知
        with self.assertRaises(ValidationError):
            DFA([0], ["a"], 0, [], [(9, "a", 0)])  # 转移源未知
        with self.assertRaises(ValidationError):
            DFA([0], ["a"], 0, [], [(0, "a", 9)])  # 转移目标未知

    def test_token_outside_alphabet_rejected(self):
        with self.assertRaises(ValidationError):
            DFA([0], ["a"], 0, [], [(0, "b", 0)])

    def test_duplicate_transition_rejected(self):
        with self.assertRaises(ValidationError):
            DFA([0, 1], ["a"], 0, [], [(0, "a", 1), (0, "a", 1)])

    def test_import_state_limit(self):
        states = list(range(DEFAULT_IMPORT_MAX_STATES + 1))
        with self.assertRaises(ValidationError):
            DFA(states, ["a"], 0, [], [])
        # 内部构造（max_states=None）不受 80 限制
        d = DFA(states, ["a"], 0, [], [], max_states=None)
        self.assertEqual(len(d.states), DEFAULT_IMPORT_MAX_STATES + 1)

    def test_alphabet_sorted_on_construction(self):
        d = DFA([0], ["b", "a"], 0, [], [])
        self.assertEqual(d.alphabet, ("a", "b"))

    def test_missing_transition_is_implicit_sink(self):
        d = DFA([0, 1], ["a", "b"], 0, [1], [(0, "a", 1)])
        self.assertTrue(d.accepts(["a"]))
        self.assertFalse(d.accepts(["b"]))       # 缺失转移 -> 汇点
        self.assertFalse(d.accepts(["a", "a"]))  # 状态 1 无出边
        self.assertFalse(d.accepts(["c"]))       # 未声明 token -> 汇点

    def test_empty_alphabet_only_empty_string(self):
        d = DFA([0], [], 0, [0], [])
        self.assertTrue(d.accepts([]))
        self.assertFalse(d.accepts(["a"]))  # 未声明 token -> 汇点


if __name__ == "__main__":
    unittest.main()
