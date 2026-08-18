# CS336 作业一 —— Transformer 基础构件实现（nn_utils.py）

日期：2026-08-18

---

## 一、今日任务

BPE tokenizer 部分（训练 + 编码解码）已经全部跑通，今天开始转向 Transformer
模型本身。按照依赖关系拆出的六阶段规划，今天做**第一阶段**：实现互相独立、
不依赖 attention/RoPE 的最底层积木——`Linear`、`Embedding`、`RMSNorm`、
`softmax`、`SwiGLU`。这些会被后面的 attention、TransformerBlock、
TransformerLM 直接调用。

---

## 二、知识点汇总（按作业文档的框架和公式）

### 1. 整体框架（3.1 Transformer LM）

语言模型接收 `(batch_size, sequence_length)` 的整数 token id 序列，输出
`(batch_size, sequence_length, vocab_size)` 的归一化概率分布（每个位置预测
下一个 token）。整体结构（对应文档 Figure 1）：

```
token ids
   │
   ▼
Token Embedding                      形状 (batch, seq_len) -> (batch, seq_len, d_model)
   │
   ▼
Transformer Block × num_layers       每个 block 输入输出形状都是 (batch, seq_len, d_model)
   │
   ▼
最终 RMSNorm（pre-norm 架构专属，用来在输出前把激活值缩放到合适范围）
   │
   ▼
Linear（"输出嵌入"/LM head）           (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
   │
   ▼
Softmax -> 下一个 token 的概率分布
```

今天写的 `Linear`、`Embedding`、`RMSNorm`、`softmax`、`SwiGLU` 对应框架图
里的红色 token embedding 块、每个 block 内部的 Norm 子层、以及 block 内部
Feed-Forward 子层（Figure 2 里 pre-norm block 的两个子层之一）。

### 2. Pre-norm Transformer block 的公式（3.4）

文档 Figure 2 描述的 pre-norm block，两个子层的更新公式：

```
z = x + MultiHeadSelfAttention(RMSNorm(x))      (对应子层 1)
y = z + FFN(RMSNorm(z))                          (对应子层 2)
```

跟原始 Transformer 论文的 "post-norm"（先算子层再归一化）相反，pre-norm 是
先归一化再进子层，输出直接残差相加。文档解释这样做能保留一条"干净的残差流"
（从输入 embedding 一路不经过任何归一化直接连到最后输出），有利于梯度传播，
是当前主流 LLM（GPT-3、LLaMA、PaLM）的标准做法。今天写的 `RMSNorm` 和
`SwiGLU`（对应 FFN）就是这两个公式里会被调用的组件，`MultiHeadSelfAttention`
留到第二阶段。

### 3. Linear：公式与内存排布（3.3.2 + 3.2.1）

文档给的公式是列向量记号：

```
y = W x                    (公式 3，W ∈ R^(d_out × d_model)，x 是列向量)
```

但文档 3.2.1 特别提醒：PyTorch/NumPy 默认是**行主序**存储，工程实践里习惯用
行向量、把 batch 维度放在最前面。这意味着如果严格套用公式 y=Wx 的写法，
实际代码里要写成 **y = x W^T**（行向量版本，公式 1）。要求存参数存
`W ∈ R^(d_out × d_model)`（不存转置），是为了跟 PyTorch 自带的 `nn.Linear`
保持一致的存储格式，方便直接用官方给的参考权重做 `load_state_dict` 对拍。

文档 3.2 节建议用 `einsum` 写这类线性变换：

```
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

`...` 表示可以有任意多个前导 batch 维度（batch、序列长度等），不需要像
`view`/`transpose` 那样手动摆弄维度顺序——"einsum 记号本身就是文档"。

### 4. 参数初始化（3.3.1）

文档给了三种参数的初始化方案，都用截断正态分布 `torch.nn.init.trunc_normal_`：

```
Linear 权重:    N(μ=0, σ² = 2/(d_in+d_out))，截断在 [-3σ, 3σ]
Embedding:      N(μ=0, σ²=1)，截断在 [-3, 3]
RMSNorm 增益:   全 1（identity，记号 𝟙）
```

文档提到 pre-norm Transformer 对初始化"异常鲁棒"（不像早期网络架构那样对
初始化极度敏感），这几套近似初始化对大多数场景够用，更精细的初始化策略
留到后续作业再讨论。

### 5. RMSNorm 公式（3.4.1，公式 4）

```
RMSNorm(a_i) = (a_i / RMS(a)) * g_i

RMS(a) = sqrt( (1/d_model) * Σ_{i=1}^{d_model} a_i^2 + ε )
```

`g_i` 是可学习的逐通道"增益"参数（一共 `d_model` 个），`ε` 是数值稳定性
超参数（常取 `1e-5`）。文档明确要求：计算前把输入 upcast 到 `torch.float32`
（防止对 `a_i` 平方求和时数值溢出），算完再 downcast 回原始 dtype：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 计算 RMSNorm
result = ...
return result.to(in_dtype)
```

### 6. Position-Wise Feed-Forward：SiLU + GLU + SwiGLU（3.4.2，公式 5-7）

**SiLU（Swish）激活函数**（公式 5）：

```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))
```

文档 Figure 3 对比了 SiLU 和 ReLU：形状相似，但 SiLU 在 0 附近是光滑的
（ReLU 在 0 处有折角）。文档特别建议这里直接用 `torch.sigmoid` 实现，
出于数值稳定性考虑。

**Gated Linear Unit（GLU）**（公式 6）：

```
GLU(x, W1, W2) = σ(W1 x) ⊙ W2 x
```

`⊙` 表示逐元素相乘。GLU 的直觉（文档引用 Dauphin et al.）是"为梯度提供一条
线性通路，同时保留非线性能力，缓解深层网络的梯度消失问题"。

**SwiGLU**（公式 7，把 SiLU 和 GLU 结合）：

```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2 (SiLU(W1 x) ⊙ W3 x)

x ∈ R^d_model，W1,W3 ∈ R^(d_ff × d_model)，W2 ∈ R^(d_model × d_ff)
```

文档给出经典取值 `d_ff = 8/3 × d_model`，实现时允许取整到附近 64 的倍数
（对硬件更友好）。文档还引用了 Shazeer 那句名言，形容这类架构选择更多是
经验驱动、缺乏严格的理论解释："We offer no explanation as to why these
architectures seem to work; we attribute their success, as all else, to
divine benevolence."（我们无法解释为什么这类架构有效；和其他一切一样，
我们把它的成功归功于神的仁慈。）

### 7. Softmax 公式与数值稳定性技巧（3.4.4，公式 10）

```
softmax(v)_i = exp(v_i) / Σ_{j=1}^{n} exp(v_j)
```

文档指出直接计算容易在 `v_i` 很大时让 `exp(v_i)` 变成 `inf`（进而
`inf/inf` 变成 `NaN`）。因为 softmax 对"给所有输入加同一个常数"具有不变性
（分子分母的 `exp(c)` 会同时约掉），标准技巧是先减去当前维度的最大值 `c`，
让新的最大值变成 0，避免 `exp` 溢出，这是后面写 attention、cross-entropy
时也会反复用到的同一个技巧。

---

## 三、代码详解

文件：`cs336_basics/nn_utils.py`

### 1. `Linear`

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.weight)
```

- `torch.empty(...)` 先分配一块未初始化的内存，再用 `trunc_normal_` 原地
  （in-place，函数名结尾的下划线是 PyTorch 的命名惯例，表示原地修改）填入
  截断正态分布采样值。
- `nn.Parameter(weight)`：把普通张量包装成"可训练参数"，这样 `optimizer`
  能找到它、`loss.backward()` 会给它算梯度。
- `forward` 里的 `einsum` 字符串 `"...i,oi->...o"`：第一个操作数 `x` 的
  维度记成 `...i`（任意前导维度 + 最后一维是输入特征 `i`），第二个操作数
  `weight` 记成 `oi`（`out_features` 在前、`in_features` 在后，对应存储
  形状），输出记成 `...o`——直接从记号上就能看出"输入的最后一维跟权重的
  第二维做内积、消掉，权重的第一维变成输出的最后一维"。

### 2. `Embedding`

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

- `self.weight[token_ids]`：PyTorch 支持用整数张量直接索引另一个张量的
  第 0 维（"高级索引"），`token_ids` 形状是 `(batch_size, sequence_length)`
  时，结果自动变成 `(batch_size, sequence_length, embedding_dim)`——不需要
  手写循环，PyTorch 底层会把这个索引操作批量向量化执行，效果等价于
  `nn.functional.embedding`。

### 3. `softmax`（独立函数，不是类）

```python
def softmax(in_features, dim):
    max_val = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_val
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)
```

- `keepdim=True`：让 `max`/`sum` 的结果保留被压缩的那一维（大小变成 1），
  这样才能直接跟原始形状的 `in_features` 做逐元素减法/除法（依赖 PyTorch
  的广播机制，形状里大小为 1 的维度会自动扩展匹配）。如果不加
  `keepdim=True`，压缩掉的维度会消失，形状对不上，广播会出错或者算出
  错误结果。

### 4. `RMSNorm`

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        result = normalized * self.weight.to(torch.float32)
        return result.to(in_dtype)
```

- `x.pow(2).mean(dim=-1, keepdim=True)`：对最后一维（`d_model` 那一维）
  求平方的均值，对应公式里的 `mean(a_i^2)`；`+ self.eps` 再开方，避免
  分母算出 0（除零错误）。
- `self.weight.to(torch.float32)`：增益参数本身可能是别的 dtype（比如
  模型整体用 `bfloat16` 训练），这里显式转成 `float32` 保证跟前面算出的
  `normalized`（已经是 float32）能正常相乘，避免 dtype 不匹配报错。
- 最后 `.to(in_dtype)` 转回输入原本的精度，这样 `RMSNorm` 对外表现就是
  "形状和 dtype 都不变，只是数值被归一化了"，可以无缝插进 Transformer
  block 的其他地方。

### 5. `silu` 和 `SwiGLU`

```python
def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)
```

- 三个子模块 `w1`/`w2`/`w3` 直接复用前面写好的 `Linear` 类构造，没有
  重新定义参数逻辑——这跟之前 `Tokenizer` 复用 `train_bpe.py` 里的
  `_merge_pair` 工具函数是同一个"不重复造轮子"的思路。
- `gate * value`：两个形状都是 `(..., d_ff)` 的张量逐元素相乘（Hadamard
  积），这就是"门控"的核心——`gate`（过了 SiLU 的分支）决定 `value`
  （没过激活函数的分支）里每个位置该保留多少信息。

### 6. `round_to_multiple_of_64`

```python
def round_to_multiple_of_64(d_ff_target: float) -> int:
    return round(d_ff_target / 64) * 64
```

单独抽成一个小函数，输入任意浮点数（比如 `8/3 * 1600 = 4266.67`），输出
四舍五入到最近的 64 的倍数（`4288`）。跟主逻辑解耦，以后如果 GPU 硬件对齐
要求变了（比如换成 128 的倍数），只需要改这一个函数。

---

## 四、测试结果

`uv run pytest -k "test_linear or test_embedding or test_rmsnorm or test_softmax_matches_pytorch or test_swiglu" -v`：

```
collected 48 items / 43 deselected / 5 selected

tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
tests/test_model.py::test_swiglu PASSED
tests/test_model.py::test_rmsnorm PASSED
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

5 passed, 43 deselected in 0.14s
```




