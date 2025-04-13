"""
层归一化模块
"""

import torch
from torch import nn


class LayerNormalization(nn.Module):
    """
    对输入张量进行归一化处理，使用 LayerNorm 技术
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        """
        :param features: 层归一化的特征维度，通常等于 d_model
        :param eps: 防止方差除零错误的小常数，默认为 1e-6
        """
        super(LayerNormalization, self).__init__()
        self.eps = eps
        # nn.Parameters: 将张量(tensor)包装为可学习的模型参数, 会自动注册到模型的 parameters() 列表中, 会被优化器识别并更新
        # 构建缩放因子和偏置项，d_model 维度的特征向量，每个特征都有一个对应的缩放因子和偏置项，所有 token 共享同一套特征维度的缩放参数
        # self.alpha.shape = self.bias.shape = (d_model,)
        self.alpha = nn.Parameter(torch.ones(features))  # 缩放因子，可学习
        self.bias = nn.Parameter(torch.zeros(features))  # 偏置项，可学习

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入特征张量，形状为 (batch_size, seq_len, d_model)
        :return: 归一化后的输入特征张量，形状为 (batch_size, seq_len, d_model)
        """

        # 对每个 batch 中的每个 token 位置(一共 seq_len 个 token)：取 d_model 维的特征向量，计算这些特征的均值和标准差
        # mean.shape = std.shape = (batch_size, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # 归一化操作的维度变化分解如下：
        """
        1. x - mean →  广播机制沿着均值 mean 的最后一个维度计算时，会虚拟扩展维度，将 1 扩展到 512 维度，但内存中不会真正复制数据
            mean.shape = (batch_size, seq_len, 1) ——> (batch_size, seq_len, d_model=512)
        2. / (std + eps)    广播机制沿着标准差 std 的最后一个维度计算时，都用此轴上的第一组值进行运算，而不是真的将其从 1 扩展到 512 维度
            std.shape = (batch_size, seq_len, 1) ——> (batch_size, seq_len, d_model=512)
        3， * self.alpha  
            self.alpha.shape = (d_model=512,) ———维度扩展———> (1, 1, d_model=512)
            广播机制沿着缩放因子 alpha 的前两个维度计算时，都用此轴上的第一组值进行运算，而不是真的在内存中将其从 1 扩展到 batch_size 和 seq_len 维度
        4. + self.bias
            self.bias.shape = (d_model=512,) ———维度扩展———> (1, 1, d_model=512)
            广播机制沿着偏置项 bias 的前两个维度计算时，都用此轴上的第一组值进行运算，而不是真的在内存中将其从 1 扩展到 batch_size 和 seq_len 维度
        """
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
