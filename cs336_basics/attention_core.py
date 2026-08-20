"""
Transformer 注意力机制（第二阶段·第一部分）：
- scaled_dot_product_attention：核心的注意力计算公式
- RotaryPositionalEmbedding（RoPE）：给 Q/K 注入相对位置信息

这两个是相对独立的基础组件，不依赖彼此，也不依赖"多头"这个概念——
一次 attention 调用、一次 RoPE 调用，操作的都是单独一组 Q/K/V。
下一步会把它们组装进 MultiHeadSelfAttention。

复用第一阶段 nn_utils.py 里的 softmax，不重复造轮子。
"""

import torch
import torch.nn as nn

from cs336_basics.nn_utils import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Q: (..., queries, d_k)
    K: (..., keys, d_k)
    V: (..., keys, d_v)
    mask: (..., queries, keys) 的布尔张量，True 表示"允许看到"，False 表示
          "禁止看到"。可以没有（不做任何屏蔽）。

    返回: (..., queries, d_v)
    """
    d_k = Q.shape[-1]

    # Q K^T：把 Q 的最后一维（d_k）和 K 的最后一维（d_k）做内积消掉，
    # 得到每个 query 对每个 key 的"相似度分数"，形状 (..., queries, keys)。
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k**0.5)

    if mask is not None:
        # mask 为 False 的位置，加上负无穷，softmax 之后这些位置的权重会
        # 变成 0——等价于"完全不让它参与加权求和"。
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)  # (..., queries, keys)

    # 用算出来的权重对 V 做加权求和：keys 这一维被消掉，d_v 保留下来。
    return torch.einsum("...qk,...kd->...qd", attn_weights, V)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE：把 Q/K 向量的每一对相邻维度看成一个二维坐标，按 token 的位置
    编号旋转一个角度，从而给模型注入"相对位置"信息。

    角度公式：theta_{i,k} = i / (Θ^((2k-2)/d)) ，i 是位置编号，k 是第几对
    维度（1..d/2），Θ 是一个超参数（作业里常取 10000）。

    cos/sin 值只跟"位置编号"和"维度"有关，跟具体输入的内容无关，所以可以
    在初始化时预先算好、缓存起来（用 register_buffer，不参与训练，
    persistent=False 表示不需要存进 checkpoint，反正随时能重新算出来）。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "RoPE 要求 d_k 是偶数（每两维一组做旋转）"
        self.d_k = d_k

        # 每一对维度 k（0 到 d_k/2 - 1）对应一个频率：
        #   freq_k = 1 / theta^(2k / d_k)
        half_dim = d_k // 2
        exponents = torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim
        inv_freq = 1.0 / (theta**exponents)  # (d_k/2,)

        # 位置编号 0..max_seq_len-1，跟频率做外积，得到每个 (位置, 维度对)
        # 组合对应的旋转角度。
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (max_seq_len, d_k/2)

        # 预先缓存 cos/sin，forward 时直接按位置切片查表，不用每次都重新算
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k) —— 要旋转的 Q 或 K 向量
        token_positions: (..., seq_len) —— 每个位置对应的位置编号
        返回: (..., seq_len, d_k)，形状不变，只是做了旋转
        """
        # 按 token_positions 从预先缓存好的表里查出对应的 cos/sin，
        # 形状变成 (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 把 x 的最后一维拆成两半：x1 是"偶数位"，x2 是"奇数位"，
        # 分别对应每一对旋转维度里的两个分量。
        x1, x2 = x[..., 0::2], x[..., 1::2]  # 各自 (..., seq_len, d_k/2)

        # 二维旋转矩阵作用在 (x1, x2) 上：
        #   x1' = x1*cos - x2*sin
        #   x2' = x1*sin + x2*cos
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 把旋转后的两半按原来交错的顺序拼回去，形状恢复成 (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_x1
        out[..., 1::2] = rotated_x2
        return out