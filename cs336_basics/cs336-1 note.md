# CS336 作业一 —— BPE Tokenizer 训练 (train_bpe.py)

日期：2026-08-15

---

## 一、知识点汇总

### 1. 为什么用字节级 BPE（Byte-level BPE）

- **传统词级/字符级分词**的问题：遇到词表外的词（OOV）只能用 `<UNK>`，会丢信息；不同语言、emoji、代码符号等很难用固定字符集覆盖。
- **字节级**：先把文本按 UTF-8 编码成字节（0~255），词表的最小单元就是 256 个字节。好处是**任何字符串都能无损表示**，不存·在 OOV。
- **BPE（Byte Pair Encoding）**：在字节序列的基础上，反复把**出现频率最高的相邻二元组（pair）合并成一个新 token**，直到词表达到目标大小。合并次数越多，词表里长 token 越多，压缩率越高。

### 2. 词表（vocab）的构成

`vocab_size` 由三部分组成：

```
256 个初始字节 token  +  特殊 token（如 <|endoftext|>）  +  合并产生的新 token
```

对应代码里：

```python
self.vocab = {i: bytes([i]) for i in range(256)}     # 256 个单字节 token
for token in special_tokens:
    self.vocab[len(self.vocab)] = token.encode("utf-8")  # 特殊 token 追加在后面
# 剩下的名额 = vocab_size - len(self.vocab)，全部留给合并产生的新 token
```

### 3. 预分词（pre-tokenization）

直接对整篇文本做字节对合并代价很大（会跨词合并出奇怪的东西，比如空格和词粘在一起）。GPT-2 的做法是先用一个正则把文本切成"预分词单元"（大致是：单词、数字、标点、空白），**合并只在每个预分词单元内部进行，不跨单元**。

```python
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

- `'(?:[sdmt]|ll|ve|re)`：英文缩写，如 `'s` `'ll` `'ve`
- ` ?\p{L}+`：可选前导空格 + 连续字母（单词）
- ` ?\p{N}+`：可选前导空格 + 连续数字
- ` ?[^\s\p{L}\p{N}]+`：可选前导空格 + 连续标点符号
- `\s+(?!\S)` / `\s+`：处理连续空白

### 4. 特殊 token 不参与合并

特殊 token（如 `<|endoftext|>`）在训练前就先从文本中**切分出去**（`re.split`），保证：
1. 预分词和合并都不会看到特殊 token 的内部结构；
2. 合并产生的新 token 不会跨越特殊 token 的边界，泄漏出 `<|` 这种字节片段。

### 5. 合并规则与打破平局（tie-break）

每一轮：
1. 统计当前所有相邻 pair 的出现次数；
2. 选出**出现次数最多**的 pair；
3. 如果并列最多，选**字节字典序更大**的那个 pair（和官方 GPT-2 实现保持一致，否则训练出来的 merges 顺序会跟参考答案对不上）。

### 6. 增量更新 —— 训练效率的关键

朴素实现：每合并一次就重新扫描全部文本统计一遍 pair 频率 —— 太慢。

更快的做法：维护一个"pair → 包含该 pair 的预分词单元下标集合"的**倒排索引**，每次合并只需要：
- 找到受影响的预分词单元（通过倒排索引，不用扫全表）；
- 只在这些单元内部减掉旧 pair 计数、做合并、加上新 pair 计数。

这是本次代码里 `pair_to_pretoken_indices` 存在的意义。

---

## 二、训练全流程

```
读取文件（bytes）
      │
      ▼
按 special_tokens 切分文本（避免合并跨越特殊 token）
      │
      ▼
对每个片段用 GPT-2 正则做预分词 → 得到若干"预分词单元"
      │
      ▼
把每个预分词单元编码成字节 id 列表  pretokens: list[list[int]]
      │
      ▼
统计所有相邻 pair 的出现次数 + 建立 pair → 预分词单元下标 的倒排索引
      │
      ▼
循环，直到达到 vocab_size 或没有 pair 可合并：
   ├─ 选出频率最高（并列则字典序最大）的 pair
   ├─ 生成新 token，加入 vocab；把这次合并记录进 merges
   ├─ 只处理受影响的预分词单元：
   │     减掉旧 pair 计数 → 执行合并 → 加上新 pair 计数
   └─ 更新对应预分词单元的 token id 序列
      │
      ▼
返回 vocab: dict[int, bytes]，merges: list[tuple[bytes, bytes]]
```

---

## 三、代码详解

文件：`cs336_basics/train_bpe.py`

### 1. `TokenizerTrainer` 的核心状态

| 属性 | 类型 | 含义 |
|---|---|---|
| `self.vocab` | `dict[int, bytes]` | token id → token 的字节内容。初始 256 项 + 特殊 token，训练过程中不断追加 |
| `self.merges` | `list[tuple[bytes, bytes]]` | **按创建顺序**记录每次合并的两个 token 的字节内容，供后续 tokenizer 编码时复用同样的合并顺序 |
| `self.special_tokens` | `list[str]` | 不参与合并的特殊 token 原始字符串 |

### 2. `_pretokenize` —— 文本 → `list[list[int]]`

```python
def _pretokenize(self, raw_bytes: bytes) -> list[list[int]]:
    text = raw_bytes.decode("utf-8", errors="ignore")
    if self.special_tokens:
        split_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        segments = re.split(split_pattern, text)          # 按特殊 token 切开
    else:
        segments = [text]

    pretokens: list[list[int]] = []
    for segment in segments:
        for match in _COMPILED_PATTERN.finditer(segment):  # GPT-2 正则预分词
            pretokens.append(list(match.group(0).encode("utf-8")))
    return pretokens
```

- **返回结构**：`pretokens` 是一个"列表的列表"。外层每一项对应一个预分词单元（比如一个单词），内层是这个单元编码成 UTF-8 后**每个字节对应的整数 id**（0~255）。
- 例如单词 `"the"` → UTF-8 字节 `b"the"` → `list(b"the")` = `[116, 104, 101]`。
- 之所以先转成 `list[int]` 而不是直接存 `bytes`，是因为后面合并时需要把两个相邻 id 换成一个新的 id（新 id 会 ≥256），`list[int]` 既能装原始字节 id，也能装合并出来的新 token id。

### 3. `_count_pairs` —— 建立频率表和倒排索引

```python
@staticmethod
def _count_pairs(pretokens):
    pair_to_pretoken_indices: defaultdict[tuple[int, int], set] = defaultdict(set)
    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for idx, token_ids in enumerate(pretokens):
        for pair in zip(token_ids, token_ids[1:]):      # 相邻两两配对
            pair_to_pretoken_indices[pair].add(idx)
            pair_counts[pair] += 1
    return pair_to_pretoken_indices, pair_counts
```

- `pair_counts: defaultdict[tuple[int,int], int]`：key 是相邻的 `(token_id_a, token_id_b)` 二元组，value 是这个 pair 在**全部文本**里出现的总次数。
- `pair_to_pretoken_indices: defaultdict[tuple[int,int], set[int]]`：key 同上，value 是一个 **下标集合**（`set[int]`），记录"哪些预分词单元（`pretokens` 里的第几项）包含这个 pair"。用 `set` 是因为一个预分词单元内同一个 pair 可能出现多次，去重后一个下标只存一份，而且合并后需要用 `.discard()` 高效删除。
- `zip(token_ids, token_ids[1:])` 是一个经典技巧：把序列和它自己错开一位 zip 起来，直接生成"相邻元素对"，不用手写下标循环。

### 4. `_pick_best_pair` —— 选出本轮要合并的 pair

```python
def _pick_best_pair(self, pair_counts):
    def rank(pair):
        return pair_counts[pair], (self.vocab[pair[0]], self.vocab[pair[1]])
    return max(pair_counts, key=rank)
```

- `rank` 返回一个**元组** `(次数, (bytes, bytes))`。Python 对元组比较是按位比较：先比第一项（次数），次数相同再比第二项（两个 token 的字节内容，按字典序）。
- `max(pair_counts, key=rank)`：`pair_counts` 是 dict，直接 `max()` 遍历的是它的 **key**（也就是所有 pair），配合 `key=rank` 就是"在所有 pair 里找 rank 元组最大的那个"。

### 5. `_merge_pair` —— 在一个预分词单元内部执行合并

```python
def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged_ids: list[int] = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
            merged_ids.append(new_id)
            i += 2                      # 命中就跳过两个位置
        else:
            merged_ids.append(token_ids[i])
            i += 1
    return merged_ids
```

- 双指针线性扫描：`i` 从头走到尾，一旦发现 `token_ids[i], token_ids[i+1]` 正好等于要合并的 `pair`，就往结果里塞入 `new_id` 并让 `i` 前进 2；否则原样塞入当前元素，`i` 前进 1。
- 输入输出都是 `list[int]`，长度只会变短（每命中一次少 1 个元素）。

### 6. `train_bpe` 主循环 —— 增量更新是效率关键

```python
for _ in range(num_merges_needed):
    if not pair_counts:
        break

    best_pair = self._pick_best_pair(pair_counts)
    new_token_id = len(self.vocab)
    new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
    self.vocab[new_token_id] = new_token_bytes
    self.merges.append((self.vocab[best_pair[0]], self.vocab[best_pair[1]]))

    affected_indices = pair_to_pretoken_indices[best_pair].copy()   # ① 只取受影响的下标
    for idx in affected_indices:
        token_ids = pretokens[idx]
        if len(token_ids) < 2:
            continue

        for pair in zip(token_ids, token_ids[1:]):                 # ② 减掉这个单元贡献的旧计数
            pair_counts[pair] -= 1
            pair_to_pretoken_indices[pair].discard(idx)
            if pair_counts[pair] == 0:
                del pair_counts[pair]
                del pair_to_pretoken_indices[pair]

        merged_token_ids = _merge_pair(token_ids, best_pair, new_token_id)  # ③ 执行合并

        for pair in zip(merged_token_ids, merged_token_ids[1:]):   # ④ 加回合并后的新计数
            pair_counts[pair] += 1
            pair_to_pretoken_indices[pair].add(idx)

        pretokens[idx] = merged_token_ids                          # ⑤ 用新序列覆盖旧序列
```

逐行对照数据结构变化：

- ①：`pair_to_pretoken_indices[best_pair]` 直接给出"哪些预分词单元受这次合并影响"，**不需要遍历全部 `pretokens`**，这是相比朴素实现的核心加速点。`.copy()` 是因为下面的循环会修改这个 set 本身（`discard`/`add`），不 copy 会边遍历边改，出错。
- ②：合并前，这个预分词单元原来贡献的所有相邻 pair 计数都要先减掉（因为合并后这些 pair 可能不再相邻或数量变了），计数归零就把这个 key 整个删掉，保持 `pair_counts` 里只留下"确实还存在"的 pair（`_pick_best_pair` 遍历的就是这个 dict 的 key）。
- ③：调用第 5 节的 `_merge_pair`，得到合并后的新 `list[int]`。
- ④：把合并后新产生的相邻 pair 计数加回去，同时更新倒排索引。
- ⑤：`pretokens[idx]` 原地替换成新序列，供下一轮合并使用。

### 7. 复杂度直觉

- 朴素实现：每轮合并都要扫一遍全部预分词单元 → 总复杂度 ≈ `O(合并次数 × 文本总长度)`。
- 本实现：每轮合并只扫"受影响的预分词单元"（通常远小于全体），均摊下来快很多。这也是训练 500 词表时能在约 0.22 秒内跑完（测试要求 1.5 秒内）的原因。

