import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    将词表索引转化为词嵌入向量
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        :params d_model: 词嵌入的维度
        :params vocab_size: 词表大小
        """
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # 创建一个形状为 (vocab_size, d_model) 的矩阵，每行对应一个词的嵌入向量，默认使用 Xavier 初始化
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :params x: 词表索引，shape=(batch_size, seq_len)
        :return 通过索引查找对应的行(嵌入向量)，shape=(batch_size, seq_len, d_model)
        这里论文中有一个小细节，就是在词嵌入向量乘以一个根号下 d_model 的系数
        """
        return self.embedding(x) * math.sqrt(self.d_model)
