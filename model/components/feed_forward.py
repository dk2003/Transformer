import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    前馈网络，用于对解码器的输出进行非线性变换。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        :param d_model: 词嵌入的维度，等于一个 token 位置编码的维度
        :param d_ff: 前馈网络的隐藏层维度，通常是 d_model 的 4 倍
        :param dropout: dropout 概率
        """
        super(FeedForwardBlock, self).__init__()
        # 第一层线性变换，输入维度为 d_model，输出维度为 d_ff
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)  # W1 * x + b1
        # Dropout 层，用于防止过拟合，位于残差连接之前
        self.dropout = nn.Dropout(dropout)
        # 第二层线性变换，输入维度为 d_ff，输出维度为 d_model
        self.linear_2 = nn.Linear(
            d_ff, d_model, bias=True
        )  # W2 * (ReLu(W1 * x + b1)) + b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入的词嵌入矩阵，形状为 (batch_size, seq_len, d_model)
        :return: 前馈网络的输出，形状为 (batch_size, seq_len, d_model)
        """
        # 形状变化：(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        # linear 会默认在最后一个维度上做线性变换
        x = self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        return x
