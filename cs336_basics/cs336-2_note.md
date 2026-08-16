# CS336 作业一 —— BPE 训练多进程优化 + Tokenizer 实现

日期：2026-08-16

---
## 一、多进程 BPE 训练优化

### 1. 为什么要优化

单进程版本的瓶颈主要在**预分词阶段**：要把整个语料文件读进内存、用正则
逐字符扫描切出所有 pretoken。这一步是"读文件 + 跑正则"，天然可以按文件区间
拆开并行处理，互不依赖——这正是适合上多进程的场景。而**合并阶段**
（不断找最高频 pair 并合并）本质上是串行的，没法并行，所以优化只做在预分词
这一半。

### 2. 整体流程

```
原始训练流程（单进程）：
  读整个文件 → 正则预分词（单进程，逐字符扫描）→ 统计 pair → 合并循环

优化后流程（多进程预分词）：
  find_chunk_boundaries 把文件切成 N 份（N = 进程数 × chunk_multiplier）
        │
        ▼
  每份的 (文件路径, start, end, special_tokens) 打包成任务
        │
        ▼
  multiprocessing.Pool（forkserver）+ imap_unordered 并行处理
        │   （每个子进程自己 seek+read 自己的区间，独立预分词，
        │    并在子进程内部就把重复 pretoken 去重计数）
        ▼
  主进程流式合并所有子进程返回的 {pretoken: count} 字典
        │
        ▼
  转成 pretokens / counts 两个并行数组，交给合并阶段（跟昨天逻辑一致）
```

### 3. 关键设计点

**(1) 切块数 ≠ 进程数**

```python
num_chunks = self.num_processes * self.chunk_multiplier  # 默认 8 倍
boundaries = find_chunk_boundaries(f, num_chunks, split_bytes)
```

如果切块数等于进程数，每块会非常大（比如 OWT 用 10 进程只切 10 份，单份超过
1GB），单个子进程处理这么大一块内存压力很大。切成比进程数多几倍的小块，配合
`imap_unordered` 让调度器动态分配（同一时间只有 `num_processes` 个任务在跑，
跑完一个立刻领下一个），既控制了单次内存峰值，又让各进程负载更均衡（避免某个
超大块拖后腿）。

**(2) worker 内部就地去重计数，减少跨进程传输量**

```python
def _pretokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")
    ...
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for segment in segments:
        for match in _COMPILED_PATTERN.finditer(segment):
            token_ids = tuple(match.group(0).encode("utf-8"))
            counts[token_ids] += 1
    return counts
```

- 每个子进程**自己** `seek + read` 自己负责的字节区间，不需要主进程把大段文本
  序列化传给它——避免了跨进程传输大字符串的开销。
- 返回的不是"每次出现存一条"的列表，而是**先在子进程内部去重计数**
  （`{pretoken_id元组: 次数}`）。高频词（比如英文的 `the`、` a`）如果不去重直接
  传回主进程，会浪费大量内存和序列化（pickle）时间——这是多进程版本相比单进程
  最关键的一处内存优化。

**(3) 主进程用 `imap_unordered` 流式合并**

```python
merged_counts: dict[tuple[int, ...], int] = defaultdict(int)
with ctx.Pool(processes=self.num_processes) as pool:
    for worker_result in pool.imap_unordered(_pretokenize_chunk, tasks):
        for token_ids, cnt in worker_result.items():
            merged_counts[token_ids] += cnt
```

`imap_unordered` 谁先算完就先把谁的结果拿回来合并，不会像 `pool.map` 那样
等全部 worker 都跑完才一次性把所有结果堆进内存——边算边合并、及时释放。

**(4) 合并阶段改成"按 count 加权"，配合 `gc.disable()`**

因为预分词阶段已经把 pretoken 去重了，原来"每次出现存一条、+1/-1"的写法要相应
改成"按这个 pretoken 的出现次数（`counts[idx]`）整体加减"：

```python
for idx in affected_indices:
    ...
    pre_count = counts[idx]
    for pair in zip(token_ids, token_ids[1:]):
        pair_counts[pair] -= pre_count
        ...
```

合并循环本身增删大量小对象（dict/set），用 `gc.disable()` 包住整个循环、
结束后再 `gc.enable(); gc.collect()`，减少垃圾回收器的干扰开销：

```python
gc.disable()
try:
    for _ in range(num_merges_needed):
        ...
finally:
    gc.enable()
    gc.collect()
```

### 4. 遇到的坑：切块锚点 token 的选择

`find_chunk_boundaries` 只接受**一个** `split_special_token`（不支持传列表），
所以最初写成"直接拿 `special_tokens[0]`"当切割锚点。这里有个隐患：如果调用方
传参顺序里第一个特殊 token 在语料里很少出现（比如传成 `["<|pad|>",
"<|endoftext|>"]`），切块函数会找不到锚点、退化成几乎没切开，起不到并行效果。
改成优先找 `<|endoftext|>`（最常见的文档分隔符），找不到才退化用列表第一个：

```python
if "<|endoftext|>" in self.special_tokens:
    split_bytes = "<|endoftext|>".encode("utf-8")
else:
    split_bytes = self.special_tokens[0].encode("utf-8")
```

---

## 二、Tokenizer 实现

### 1. 设计目标
逻辑上完全对应课程文档
（`the` → 拆字节 → 按 merges 顺序逐条尝试合并 → 查表转 id），，靠**pretoken 级别缓存**弥补重复计算的开销。
### 2. 编码流程

```
输入文本
   │
   ▼
按特殊 token 正则切分（特殊 token 长度降序排序，避免短 token 抢先匹配）
   │
   ├── 特殊 token 片段 → 直接查表转成一个 id
   │
   └── 普通片段 → GPT-2 正则预分词，得到若干 pretoken
                       │
                       ▼
                 每个 pretoken：
                   - 先查缓存，命中直接返回
                   - 没命中：拆成单字节 bytes 列表
                     → for merge_pair in self.merges:
                         按训练时的顺序逐条尝试合并，
                         合并成一个 token 就提前 break
                     → 每个字节片段查 token_to_id 转成 id
                     → 结果存入缓存
   │
   ▼
把所有片段的 id 按原文顺序拼接，得到最终 token id 列表
```

### 3. 关键代码

**(1) 单个 pretoken 的合并（对应课程文档"处理预分词 the"的例子）**

```python
def _encode_one_pretoken(self, pretoken: str) -> list[int]:
    if pretoken in self._cache:
        return self._cache[pretoken]

    parts: list[bytes] = [bytes([b]) for b in pretoken.encode("utf-8")]

    for merge_pair in self.merges:
        if len(parts) == 1:
            break
        parts = self._apply_one_merge(parts, merge_pair)

    token_ids = [self.token_to_id[part] for part in parts]
    self._cache[pretoken] = token_ids
    return token_ids
```

- 第一步：把 pretoken 编码成 UTF-8 字节，每个字节单独当一个 `bytes` 对象放进
  `parts` 列表（比如 `"the"` → `[b't', b'h', b'e']`）。
- 第二步：严格按 `self.merges` 记录的顺序（也就是训练时的合并顺序）逐条尝试
  应用。这一点很关键——**编码时必须复用训练时的合并顺序**，才能保证结果和
  训练阶段的行为一致（比如训练时先合并 `(t,h)` 再合并 `(th,e)`，编码时也要
  按这个顺序试，不能先试 `(th,e)`）。
- 加了 `if len(parts) == 1: break`：一旦合并成一个 token，后面的规则不用再试。

**(2) 单次合并（和昨天 `train_bpe.py` 里 `_merge_pair` 逻辑一致，只是操作对象
从 `int` id 换成 `bytes`）**

```python
@staticmethod
def _apply_one_merge(parts: list[bytes], merge_pair: tuple[bytes, bytes]) -> list[bytes]:
    new_parts: list[bytes] = []
    i = 0
    while i < len(parts):
        is_match = i < len(parts) - 1 and (parts[i], parts[i + 1]) == merge_pair
        if is_match:
            new_parts.append(parts[i] + parts[i + 1])
            i += 2
        else:
            new_parts.append(parts[i])
            i += 1
    return new_parts
```

双指针线性扫描，跟训练阶段的 `_merge_pair` 是完全一样的思路，只是这里拼接的
是 `bytes + bytes`（字节串拼接）而不是替换成一个新的整数 id。

**(3) 特殊 token 切分**

```python
def encode(self, text: str) -> list[int]:
    if self.special_pattern is None:
        return self._encode_without_special_tokens(text)

    ids: list[int] = []
    pos = 0
    for match in self.special_pattern.finditer(text):
        normal_text = text[pos:match.start()]
        if normal_text:
            ids.extend(self._encode_without_special_tokens(normal_text))
        special_token_bytes = match.group().encode("utf-8")
        ids.append(self.token_to_id[special_token_bytes])
        pos = match.end()

    remaining_text = text[pos:]
    if remaining_text:
        ids.extend(self._encode_without_special_tokens(remaining_text))
    return ids
```

用 `finditer` 找出文本里所有特殊 token 出现的位置，特殊 token 之间/前后的
"普通文本"分别走正常的预分词+合并流程，特殊 token 本身直接查表映射成一个 id，
不会被拆开。

**(4) `encode_iterable`：简化成按行迭代**

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for chunk in iterable:
        for token_id in self.encode(chunk):
            yield token_id
```

假设传进来的 `iterable`（通常是打开的文件对象，`for line in f` 天然按行迭代）
产出的每一块内容本身就是完整的一行，不会把单词或特殊 token 硬切成两半——这是
文本文件按行读取的自然特性，所以不需要额外写"防止切碎"的缓冲区逻辑，是牺牲了
一点点通用性（万一调用方传入的不是按行分割的可迭代对象，理论上有切碎风险）
换来的极简实现。

**(5) `decode`**

```python
def decode(self, ids: list[int]) -> str:
    all_bytes = b"".join(self.vocab[token_id] for token_id in ids)
    return all_bytes.decode("utf-8", errors="replace")
```

按 id 查表拿到每个 token 的字节内容，拼接后用 `errors="replace"` 解码——
遇到无法解码的字节（用户可能传入任意整数 id 序列）用 U+FFFD 替代，不报错。


## 三、实验结果

### 1. 运行环境

- 集群：NSCC ASPIRE2A（新加坡国立超算中心）
- 节点：Standard Compute Node，Dual-CPU AMD EPYC 7713，128 物理核 / 512GB 内存
- 通过 PBS 提交任务，队列 `normal`

### 2. TinyStories 训练（10K 词表）

| 指标 | 结果 |
|---|---|
| 语料大小 | 2.1 GB |
| 目标词表大小 | 10000 |
| 进程数 | 32 |
| 训练耗时 | **87.1 秒** |
| 实际词表大小 | 10000（256 字节 + 1 特殊 token + 9743 次合并） |
| 最长 token | `b' accomplishment'` |

`accomplishment` 这个词在 TinyStories（面向儿童的合成故事语料）里出现频率
极高，被完整合并成了一个 token，符合预期——跟参考资料里其他人复现的结果一致
（都是这个词），只是这次多进程 + Lustre 并行文件系统跑得明显更快
（参考资料里单进程版本要 234 秒）。

### 3. OpenWebText 训练（32K 词表）

| 指标 | 结果 |
|---|---|
| 语料大小 | 12 GB |
| 目标词表大小 | 32000 |
| 进程数 | 128 |
| 训练耗时 | 待补充（任务提交后台运行中，未确认最终结果） |
| 最长 token | 待补充 |

任务已通过 PBS 提交（`select=1:ncpus=128:mem=440G`，队列 `normal`），跑完后
产物存在 `out/owt/vocab.pkl`、`out/owt/merges.pkl`，日志在 `owt_out.log`。
下次继续补上具体耗时和最长 token 的结果。

### 4. Tokenizer 正确性测试

`uv run pytest tests/test_tokenizer.py -v`：
已经全部pass
---

