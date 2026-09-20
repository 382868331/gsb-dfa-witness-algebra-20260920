# 有限自动机的语言代数与最短见证

对显式给出的确定有限自动机（DFA）做语言运算与比较，不解析正则表达式。
纯 Python 3.14 标准库，Windows 原生可运行，无第三方依赖。

## 运行

```bash
python demo.py                              # 演示：正常结果 + 实际触发的失败
python -m unittest discover -s tests -v     # 测试
```

## 接口（`dfalib.py`）

### 构造

```python
DFA(states, alphabet, start, accepting, transitions, *, max_states=80)
```

- `states`：整数状态 id 集合（`bool` 不算）；`start` 必须在其中。
- `alphabet`：字符串 token 集合，最多 8 个，构造后按字典序排序存储。
- `accepting`：接受态集合，必须是 `states` 的子集。
- `transitions`：`(src, token, dst)` 三元组的可迭代对象，可以不完整；
  缺失转移进入隐式拒绝汇点。
- 未知状态引用、重复字母、重复转移、非整数状态 id、字母表外 token
  一律抛 `ValidationError`。
- 外部导入的原始 DFA 最多 80 个状态（`max_states` 默认值）；
  库内运算结果用 `max_states=None` 构造，不受此限。

`dfa.accepts(tokens)` 按隐式汇点语义运行；未声明的 token 同样进入汇点。

### 语言运算

```python
intersect(a, b, *, limit=10000)   # 交
union(a, b, *, limit=10000)       # 并
difference(a, b, *, limit=10000)  # 差 L(a) \ L(b)
complement(dfa, *, limit=10000)   # 补（相对自身显式字母表；空字母表下只有空串）
minimize(dfa, *, limit=10000)     # 最小化
```

- 二元运算先把双方字母表统一为并集（字典序），原机未声明的 token 进入汇点。
- 单次运算最多探索 `limit` 个可达状态（默认 10000）：实际探索达到上限后
  仍需新状态时抛 `StateLimitError`；不按笛卡尔积总大小预先拒绝稀疏可达乘积。
- `minimize` 返回 `Minimized(dfa, state_map, sink_state)`：
  - `dfa`：只保留可达状态、分割细化（Hopcroft）产生的真正最小完备 DFA，
    从初态 0 开始按 token 顺序 BFS 重新编号；
  - `state_map`：原可达状态 -> 最小化状态 id（不可达状态不出现）；
  - `sink_state`：原机隐式汇点可达时它在最小化机中的状态 id，否则 `None`。

### 等价 / 包含判断

```python
check_equivalent(a, b, *, limit=10000)  # -> EquivalenceResult
check_included(a, b, *, limit=10000)    # -> InclusionResult，L(a) ⊆ L(b)
are_equivalent(a, b) / is_included(a, b)  # 布尔便捷封装
```

- 在可达乘积图上 BFS（不按固定长度枚举），失败时返回最短见证 token 列表，
  同长度按 token 序列字典序最小。
- **空列表 `[]` 是合法见证**（双方初态接受性不同），与"不存在见证"的
  `None` 严格区分；判断时请用 `result.equivalent` / `result.contained`
  或 `witness is None`。
- 见证返回前在双方上用 `accepts` 实际复核，接受性记入
  `accepted_by_a` / `accepted_by_b`。

## 输入限制汇总

| 项目 | 限制 |
|---|---|
| 字母表 | 最多 8 个不同的字符串 token |
| 导入的原始 DFA | 最多 80 个状态 |
| 单次运算可达状态 | 默认 10000，超出抛 `StateLimitError` |
| 状态 id | 必须是整数 |

## 文件

- `dfalib.py` — 核心库
- `samples.py` — 演示/测试共用样例（后缀自动机、模计数器、链）
- `demo.py` — 演示：等价判定与最小化（正常）、不等价最短见证（失败一）、
  状态上限实际触发（失败二）
- `tests/` — 校验、运算交叉核验、最小化可区分性、见证、状态上限测试
