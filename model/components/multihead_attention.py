import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    """
    多头注意力机制，用于对输入序列进行加权聚合，得到一个新的序列。
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        :param d_model: 词嵌入的维度，等于一个 token 位置编码的维度
        :param h: 多头注意力的头数，论文中 h=8
        :param dropout: dropout 概率
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        # 每个头的维度，论文中 d_q = d_k = d_v = d_model / h
        self.d_k = d_model // h

        # 将所有 h 个头一起进行线性变换，而不是每个头单独进行线性变换，这样
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # 输入 Q 的线性变换: Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # 输入 K 的线性变换: Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # 输入 V 的线性变换: Wv

        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出的线性变换: Wo

        # 残差 dropout：在进行残差连接之前，先对主干输出进行 dropout
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        """
        单头缩放点积注意力
        :param query: 输入的查询向量，形状为 (batch_size, h, seq_len, d_k)
        :param key: 输入的键向量，形状为 (batch_size, h, seq_len, d_k)
        :param value: 输入的值向量，形状为 (batch_size, h, seq_len, d_k)
        :param mask: mask，编码器掩码的形状为 (batch_size, 1, 1, seq_len)，解码器掩码的形状为 (batch_size, 1, seq_len, seq_len)
        :param dropout: dropout 层
        :return: 单头注意力汇聚结果，形状为：(batch_size, h, seq_len, d_k)；单头注意力权重，形状为 (batch_size, h, seq_len, seq_len)
        """
        d_k = query.shape[-1]
        # 计算单头注意力分数，query.shape=(batch_size, h, seq_len, d_k), 交换维度方便矩阵乘法 key.transpose(-2, -1).shape=(batch_size, h, d_k,seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # attention_scores.shape = (batch_size, h, seq_len, seq_len)
        if mask is not None:
            # 应用 mask，masked_fill_ 将 mask 中的 0 替换为 -1e9，这样在 softmax 时，这些位置的注意力分数就会变为 0，相当于屏蔽了这些位置的信息
            # 1. 当编码器 mask 形状为 (batch_size, 1, 1, seq_len) 时：在头维度(h)和第二个序列长度维度上逻辑扩展,实际比较时会变成 (batch_size, h, seq_len, seq_len)
            # 2. 当解码器 mask 形状为 (batch_size, 1, seq_len, seq_len) 时：仅在头维度(h)上逻辑扩展,变成 (batch_size, h, seq_len, seq_len)
            """
            千万要注意，masked_fill_ 是原地操作，mask_fill 不是！！！！，这里不使用原地操作的话，掩码根本不会起作用
            """
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 对注意力分数进行 softmax，得到注意力权重
        attention_weights = attention_scores.softmax(
            dim=-1
        )  # shape = (batch_size, h, seq_len, seq_len)
        if dropout is not None:  # 根据原始论文，此处应该是不需要 dropout 的
            # 对注意力权重进行 dropout，防止过拟合
            attention_weights = dropout(attention_weights)

        # 对注意力权重和值向量进行加权求和，得到输出向量
        # 这里用到了批量矩阵乘法：attention_weights.shape(batch_size, h, seq_len, seq_len) * value.shape(batch_size, h, seq_len, d_k) = (batch_size, h, seq_len, d_k)
        # 批量矩阵乘法：torch.bmm 和 @，对于 (B,N,M) 和 (B,M,K) 的矩阵，二者行为相同，输出为 (B,N,K) 的矩阵。对于四维的 (B,H,N,M) 和 (B,H,M,K) 的矩阵，@ 可以直接处理，而前者需要将输入转化为三维，运算之后将结果转回四维。
        return attention_weights @ value, attention_weights

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        :param q: 输入的查询向量，形状为 (batch_size, seq_len, d_model)
        :param k: 输入的键向量，形状为 (batch_size, seq_len, d_model)
        :param v: 输入的值向量，形状为 (batch_size, seq_len, d_model)
        :param mask: mask，形状为 解码器：(batch_size, h=1, seq_len, seq_len) 或者 编码器：(batch_size, h=1, 1, seq_len)
        :return: 输出的向量，形状为 (batch_size, seq_len, d_model)
        """
        query = self.w_q(
            q
        )  # (batch_size, seq_len, d_model) ——> (batch_size, seq_len, d_model)
        key = self.w_k(
            k
        )  # (batch_size, seq_len, d_model) ——> (batch_size, seq_len, d_model)
        value = self.w_v(
            v
        )  # (batch_size, seq_len, d_model) ——> (batch_size, seq_len, d_model)

        # 对输入进行线性变换为 (batch_size, seq_len, d_model) 后，将 query, key, value 从嵌入维度拆分为 h 个头，形状为 (batch_size, seq_len, h, d_k)，其中 d_k = d_model / h
        # 然后交换 h 和 seq_len 的位置，之后的形状为 (batch_size, h, seq_len, d_k)，每个句子会被输入 h 个不同的注意力头进行注意力汇聚，每个头都可以看到整个句子
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # 计算多头注意力汇聚结果和注意力权重
        # 形状为 x.shape=(batch_size, h, seq_len, d_k) 和 attention_weights.shape=(batch_size, h, seq_len, seq_len)
        x, self.attention_weights = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # 将 h 个头的输出拼接起来(concat)而非相加
        # (batch_size, h, seq_len, d_k) ——> (batch_size, seq_len, h, d_k) ——> (batch_size, seq_len, h*d_k=d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # .contiguous() 确保内存连续，view 重组张量时不复制数据（高效内存使用）

        # 对输出进行线性变换：(batch_size, seq_len, d_model)——>(batch_size, seq_len, d_model)
        return self.w_o(x)
