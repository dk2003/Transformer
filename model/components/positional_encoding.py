import torch
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    三角位置编码，用于将位置信息加入到词嵌入向量中，
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        """
        :param d_model: 词嵌入的维度，等于一个 token 位置编码的维度
        :param seq_len: 序列长度, 不同的位置嵌入不同的位置信息
        :param dropout: dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # 创建一个 Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵，形状为 (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # 创建一个位置向量，表示从 0~seq_len 的每个位置，可以看作是 pos/10000^(2i/d_model) 中的分子 ps
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # shape: (seq_len,)———unsqueeze———>(seq_len, 1)
        # 创建分母的频率向量: 10000^(2i/d_model)，如下的实现可以保证数值稳定性具体实现参见 readme 文档
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # shape=(d_model/2,)

        # 正弦函数用于每个位置的位置向量中的偶数元素。取所有行( : ), 从第 0 列开始，每隔 2 列取一列( 0::2 )
        # 广播: position 被逻辑扩展为(seq_len, 1)——>(seq_len, d_model//2)，div_term 先被实际扩展出一个额外的维度 (d_model/2,)——>(1, d_model//2)，然后被逻辑扩展到 (seq_len, d_model//2)
        pe[:, 0::2] = torch.sin(position * div_term)  # shape = (seq_len, d_model//2)
        # 余弦函数用于每个位置的位置向量中的奇数元素。取所有行( : ), 从第 1 列开始，每隔 2 列取一列( 0::2 )
        pe[:, 1::2] = torch.cos(position * div_term)  # shape = (seq_len, d_model//2)
        # 关于广播的细节是如何进行的，参考：https://ktv422gwig.feishu.cn/wiki/WUVBw8UX0iLP8Ck05AOcF0pSnve?fromScene=spaceOverview#share-ZB8fdQ8tLoDKmjxjhcWcGMYMn9g

        # 在最前面添加一个额外的维度，用于在 batch 维度上进行广播
        pe = pe.unsqueeze(0)  # shape=(seq_len, d_model)——>(1, seq_len, d_model)

        # 将位置编码矩阵注册为模型的参数， 但不会被作为可训练参数(即不会在反向传播时更新)，这一点和 nn.Parameters() 不一样
        # 缓冲区会随模型一起保存( state_dict )和加载, 自动转移到与模型相同的设备(CPU/GPU), 在模型评估和训练时保持固定不变
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入的词嵌入矩阵，形状为 (batch_size, seq_len, d_model)
        :return: 输入词嵌入矩阵加上位置编码后的结果，形状为 (batch_size, seq_len, d_model)
        """

        # 请注意：self.pe[:, : x.size(1), :] 是一个视图操作，共享底层数据，仅仅改变形状，视图操作本身不会改变原始张量的 requires_grad 属性
        # 显式调用 .requires_grad_(False) 是防御性编程，确保万无一失，位置编码不可学习
        # self.pe[:, :x.size(1), :].shape = (1, seq_len, d_model)，与 x 相加时，会自动进行广播，逻辑扩展为 (batch_size, x_seq_len, d_model) 进行相加
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        # 原始论文的一个实现细节：在编码器和解码器中，对词嵌入与位置编码的总和应用 Dropout
        return self.dropout(x)
