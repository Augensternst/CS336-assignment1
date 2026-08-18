"""
Transformer 基础构件：Linear、Embedding、RMSNorm、softmax、SwiGLU。
"""
import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    不带 bias 的线性层：y = x @ W^T

    权重存成 W（形状 (out_features, in_features)），不存 W 的转置——这是作业
    明确要求的存储方式，跟 PyTorch 自带的 nn.Linear 一致，方便后面直接用
    state_dict 加载官方给的参考权重做测试对拍。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        # 初始化：N(0, std^2)，std^2 = 2 / (d_in + d_out)，截断在 [-3std, 3std]
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        # y = x @ W^T，用 einsum 写清楚每个维度的含义，避免矩阵乘法维度对不上时
        # 出的错误信息很难看懂。"..." 表示可以有任意多个前导 batch 维度。
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    """
    词嵌入查表：给定 token id，返回对应的嵌入向量。

    权重矩阵形状 (num_embeddings, embedding_dim)，d_model（embedding_dim）放在
    最后一维，这样 forward 直接用整数张量索引就能拿到对应行。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)

        # 初始化：N(0, 1)，截断在 [-3, 3]
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...,) 整数张量 -> (..., embedding_dim)
        # 直接用整数张量索引权重矩阵的第 0 维，PyTorch 会自动按 token_ids 的
        # 形状"广播式"取出对应的行，等价于 nn.Embedding 的查表行为。
        return self.weight[token_ids]


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    对指定维度做 softmax，减去最大值防止 exp 溢出（数值稳定性技巧）。

    softmax 对"给所有元素加同一个常数"是不变的，所以减去每行的最大值不会
    改变结果，却能避免 exp(很大的数) 变成 inf。
    """
    max_val = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_val
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    跟标准 LayerNorm 的区别：不减均值，只用均方根（RMS）做缩放，再乘一个
    可学习的逐通道增益 g。计算过程要先转成 float32 再算，算完转回原来的
    dtype——因为要对激活值平方求和，如果输入本来就是 float16/bfloat16，
    平方运算容易溢出，所以先升精度保证数值稳定，算完再降回去。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 增益参数初始化成全 1（对应作业里"RMSNorm: 单位矩阵/全1"的初始化要求），
        # 也就是刚开始训练时 RMSNorm 近似什么都不做，只做纯粹的归一化。
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) -> (..., d_model)，形状不变
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS(a) = sqrt(mean(a^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x / rms

        result = normalized * self.weight.to(torch.float32)

        # 算完再转回输入原本的 dtype
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU（也叫 Swish）激活函数：SiLU(x) = x * sigmoid(x)

    用 torch.sigmoid 而不是自己手写 1/(1+exp(-x))，是作业里明确建议的写法，
    PyTorch 内置的 sigmoid 实现对数值稳定性做了处理，比手写版本更可靠。
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络：FFN(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x))

    三个不带 bias 的线性层：
      - W1、W3：把输入从 d_model 投影到 d_ff（两条并行的"门控"分支）
      - W2：把 SiLU(W1x) 和 W3x 逐元素相乘后的结果，再投影回 d_model

    d_ff 按作业要求取约 8/3 * d_model，再向上取整到 64 的倍数（对 GPU 计算
    更友好）。这里把这个取整逻辑做成一个独立的辅助函数，方便复用/测试。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) -> (..., d_model)
        gate = silu(self.w1(x))       # (..., d_ff)
        value = self.w3(x)            # (..., d_ff)
        return self.w2(gate * value)  # (..., d_model)


def round_to_multiple_of_64(d_ff_target: float) -> int:
    """
    把目标 d_ff 数值四舍五入到最近的 64 的倍数（GPU 上做矩阵乘法时，维度是
    64 的倍数能更好利用硬件的张量核心）。作业里 d_model=1600 时，
    8/3 * 1600 = 4266.67，取整到最近的 64 倍数就是 4288。
    """
    return round(d_ff_target / 64) * 64