import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    投影层，将解码器的输出进行投影变换，得到输出的向量
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        :param d_model: 每个词向量的维度
        :param vocab_size: 词表的大小
        """
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入的向量，形状为 (batch_size, seq_len, d_model)
        :return: 输出的向量，形状为 (batch_size, seq_len, vocab_size)
        """

        # 在最后一维进行 softmax ，得到每个词的概率，再进行 log ，保证数值稳定性
        return self.proj(x)
