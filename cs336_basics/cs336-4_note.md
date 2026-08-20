## CS336 作业一 —— Attention 核心组件（scaled_dot_product_attention + RoPE）

日期：2026-08-19

---

## 一、今日目标

第一阶段的基础构件（`Linear`/`Embedding`/`RMSNorm`/`softmax`/`SwiGLU`）已经全部写完测过。今天进入**第二阶段的前半部分**：实现两个相对独立、互不依赖的注意力相关组件——`scaled_dot_product_attention`（缩放点积注意力）和 `RotaryPositionalEmbedding`（RoPE，旋转位置编码）。

---

## 二、知识点汇总（按作业文档的框架和公式）

### 1. Scaled Dot-Product Attention（3.4.4，公式 10、11）

softmax 公式（公式 10）：

```py
softmax(v)_i = exp(v_i) / Σ_{j=1}^{n} exp(v_j)
```

这个已经在第一阶段写过了，今天直接复用。

Attention 公式（公式 11）：

```python
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Q ∈ R^(n × d_k)，K ∈ R^(m × d_k)，V ∈ R^(m × d_v)
```

直觉：`Q`（"我想找什么"）跟 `K`（"每个位置有什么"）做点积，得到"每个位置该给多少关注度"的分数；除以 `sqrt(d_k)` 是缩放，防止维度越高、点积期望值越大，导致 softmax 前的分数差距过大、退化成几乎只关注一个位置。算出关注度权重后，拿去加权求和 `V`（"每个位置实际携带的信息"）。

**Masking**（文档同一小节）：mask 矩阵 `M ∈ {True, False}^(n×m)`，`True` 表示"query i 允许看到 key j"，`False` 表示不允许。文档提醒：比起真的把不该看的部分从序列里切掉、单独计算，**用 mask 屏蔽效率更高**——直接在 softmax 之前，把 mask 为 `False` 的位置的分数设为负无穷，softmax 之后这些位置的权重会自动变成 0（不用改变张量形状，一次性批量算完整个序列）。

### 2. RoPE 公式（3.4.3）

对第 `i` 个位置的 query 向量 `q^(i)`，作用一个旋转矩阵 `R_i`，得到 `q'^(i) = R_i q^(i)`。`R_i` 是一个分块对角矩阵，每一块 `R_i^k`（`k` 从 1 到 `d/2`）是一个二维旋转矩阵：

```py
R_i^k = [ cos(θ_{i,k})   -sin(θ_{i,k}) ]
        [ sin(θ_{i,k})    cos(θ_{i,k}) ]

θ_{i,k} = i / Θ^((2k-2)/d)
```

`i` 是位置编号，`k` 是第几对旋转维度，`Θ` 是超参数（作业里常取 10000）。把 query/key 向量的每两个相邻维度看成一个二维坐标点，按照这个 token 在序列里的位置编号，把这个点"旋转"一个角度——位置越靠后，旋转角度越大。两个位置的 Q、K 做点积时，旋转角度的差值天然编码了"这两个位置相隔多远"（相对位置），而不只是各自的绝对位置。

文档特别强调了两个工程要点：不需要真的构造出完整的 `d × d` 矩阵（那样既费内存又费计算），应该利用这个矩阵是"分块对角、每块只是简单的 2×2 旋转"的结构，直接对向量做逐元素运算；`cos(θ_{i,k})`、`sin(θ_{i,k})` 只跟位置 `i` 和维度 `k` 有关，跟输入内容无关，同一个 RoPE 模块可以被所有层、所有 batch 复用——建议用 `self.register_buffer(persistent=False)` 预先算好、缓存起来，而不是每次 `forward` 都重新算一遍三角函数。

---

## 三、代码详解

文件：`cs336_basics/attention_core.py`

![image-20260820205904058](/Users/yang/Library/Application Support/typora-user-images/image-20260820205904058.png)

![image-20260820210000873](/Users/yang/Library/Application Support/typora-user-images/image-20260820210000873.png)

### 1. `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kd->...qd", attn_weights, V)
```

`einsum("...qd,...kd->...qd")`：`...` 表示任意前导 batch 维度（可能包含多头这一维，虽然今天写的这个函数本身不关心是不是"多头"，`...` 会自动把这些维度当成批量维度处理）。`Q` 的最后一维标成 `d`，`K` 的最后一维也标成 `d`——两个 `d` 相同意味着这一维会被内积消掉，剩下 `q`（queries 数）和 `k`（keys 数）分别保留，输出形状 `(..., q, k)`，正好是"每个 query 对每个 key 的分数矩阵"。

`scores.masked_fill(~mask, float("-inf"))`：`~mask` 是取反（`True`/`False` 互换），把"不允许看到"的位置（原来 mask 是 `False` 的地方，取反后变 `True`）填成负无穷。这里复用了第一阶段写的 `softmax` 函数，不重复实现数值稳定性技巧。

第二个 `einsum("...qk,...kd->...qd")`：这次 `k`（keys 数）是两个操作数共有的维度，会被消掉；`attn_weights` 的 `q` 和 `V` 的 `d`（这里是 d_v）被保留，得到最终输出 `(..., q, d_v)`。

### 2. `RotaryPositionalEmbedding.__init__`：预计算 cos/sin 表

```python
half_dim = d_k // 2
exponents = torch.arange(0, half_dim) / half_dim
inv_freq = 1.0 / (theta**exponents)          # (d_k/2,)

positions = torch.arange(max_seq_len)
angles = torch.einsum("i,j->ij", positions, inv_freq)  # (max_seq_len, d_k/2)

self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
```

`inv_freq`：对应公式里 `1/Θ^((2k-2)/d)` 这部分，跟位置 `i` 无关，先单独算出"每一对维度的旋转频率"。

`einsum("i,j->ij", positions, inv_freq)`：这是一个**外积**——`positions` 形状 `(max_seq_len,)`，`inv_freq` 形状 `(d_k/2,)`，两者不共享任何字母（`i` 和 `j` 都各自独立保留在输出里），结果就是把两个一维向量的所有两两组合都算出来，形状变成 `(max_seq_len, d_k/2)`——第 `i` 行第 `k` 列正好就是 `θ_{i,k}`，跟公式完全对应。

`register_buffer(..., persistent=False)`：把 `cos_cached`/`sin_cached` 注册成模块的"缓冲区"（buffer）而不是"参数"（parameter）——缓冲区会跟着模块一起被 `.to(device)` 移动到正确的设备上，但不会被优化器更新、也不会出现在梯度计算里；`persistent=False` 表示存 checkpoint 时不需要把这张表也存进去（反正下次加载模型重新算一遍就有了，没必要占存储空间）。

### 3. `RotaryPositionalEmbedding.forward`：应用旋转

```python
def forward(self, x, token_positions):
    cos = self.cos_cached[token_positions]
    sin = self.sin_cached[token_positions]

    x1, x2 = x[..., 0::2], x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    out = torch.empty_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out
```

`self.cos_cached[token_positions]`：用整数张量索引，按每个 token 实际所在的位置编号，从预先算好的表里查出对应的 `cos`/`sin` 值——复用了第一阶段 `Embedding` 类里"用整数张量做索引查表"的同一个技巧。

`x[..., 0::2]`/`x[..., 1::2]`：切片取偶数位/奇数位，把最后一维 `d_k` 拆成两半，分别对应公式里旋转矩阵作用的那一对分量 `(x1, x2)`。`rotated_x1`/`rotated_x2` 两行代码直接照抄二维旋转矩阵的展开公式，逐元素运算，不需要构造出完整的旋转矩阵再做矩阵乘法（文档提到的效率优化点）。

`torch.empty_like(x)` + 按步长切片写回：把旋转后的两半按原来交错的顺序重新拼回去（`out[..., 0::2]` 放 `x1'`，`out[..., 1::2]` 放 `x2'`），保证输出形状跟输入完全一致。

---

## 四、测试结果

`uv run pytest -k "test_scaled_dot_product_attention or test_4d_scaled_dot_product_attention or test_rope" -v`：

```
collected 48 items / 45 deselected / 3 selected

tests/test_model.py::test_scaled_dot_product_attention PASSED
tests/test_model.py::test_4d_scaled_dot_product_attention PASSED
tests/test_model.py::test_rope PASSED

3 passed, 45 deselected in 0.16s
```

**3 个全部通过**，包括三阶张量和四阶张量两种输入形状的 attention 测试都过了，说明 `einsum` 里 `...` 通配任意前导维度的写法确实按预期工作，不需要针对不同的 batch 维度数量写不同的分支逻辑。adapter 接入这次也是脚本一次跑对，没有额外调试。
