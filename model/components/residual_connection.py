import torch
import torch.nn as nn
from .layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    残差连接模块
    """

    def __init__(self, features: int, dropout: float) -> None:
        """
        :param dropout: 主干路径在进行残差连接之前，进行 dropout
        :param d_model: dimension of model
        """
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        :param x: 输入的向量，形状为 (batch_size, seq_len, d_model)
        :param sublayer: 残差连接的主干路径
        :return: 残差连接并且归一化后的向量，形状为 (batch_size, seq_len, d_model)
        """
        # 原始论文是按照下面的顺序进行的
        # return self.norm(x + self.dropout(sublayer(x)))

        # 但是很多实现都是按照下面的顺序进行的，这种顺序可以对嵌入层的输出也做一次归一化，但是对于残差连接的最终输出需要在下一个残差连接的输入中做归一化
        # 如果没有下一个残差连接（也就是编码器或者解码器的输出），那么需要对编/解码器的最终输出做一次归一化
        return x + self.dropout(sublayer(self.norm(x)))
