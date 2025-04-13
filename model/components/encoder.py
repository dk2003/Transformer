import torch
import torch.nn as nn
from .residual_connection import ResidualConnection
from .multihead_attention import MultiHeadAttentionBlock
from .feed_forward import FeedForwardBlock
from .layer_normalization import LayerNormalization


class EncoderBlock(nn.Module):
    """
    编码器块, 只包含自自注意力层和前馈层，没有使用交叉注意力
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        """
        :param self_attention_block: 自注意力层
        :param feed_forward_block: 前馈层
        :param dropout: 丢弃率
        """
        super(EncoderBlock, self).__init__()
        # 自注意力层，句子在观察自己，q、k、v 三个矩阵相同。
        # 一个句子中的每个词都在与同一句子中的其他词进行交互。
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: 编码器块的输入，可能是源序列或上一个编码器块的输出，形状为 (batch_size, seq_len, d_model)
        :param src_mask: 源序列的 mask，掩盖 <pad> 的权重
        :return: 输出
        """

        # 第一个残差块，主干路径是自注意力层
        # 传入 residual_connections 前向传播的两个参数：x 和一个表示 sublayer 的 lambda 函数
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, src_mask),
        )
        # 第二个残差块，主干路径是前馈层
        # 此处无需 lambda 表达式
        x = self.residual_connections[1](x, self.feed_forward_block)

        # 输出，形状为 (batch_size, seq_len, d_model)
        return x


class Encoder(nn.Module):
    """
    编码器，由多个编码器块组成，每个编码器块包含自注意力层和前馈层
    """

    def __init__(
        self,
        features: int,
        layers: nn.ModuleList,
    ):
        """
        :param features: 词向量维度，即 d_model
        :param layers: 多个编码器块组成的列表
        """
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入，形状为 (batch_size, seq_len, d_model)
        :param mask: mask，掩盖 <pad> 的权重，形状为 (batch_size, 1, 1, seq_len)
        :return: 输出，形状为 (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)

        # 在最后一个编码器块之后，对输出进行归一化之后，送入解码器的交叉注意力层。这里原始论文好像没有这个步骤，但是加上也问题不大
        # 在 Transformer 原始论文中，只有最后一个编码器块的输出 会被送到解码器的交叉注意力层。
        """
        - 解码器的每一层都会接收相同的编码器最终输出
        - 不是每个编码器层对应一个解码器层
        - 编码器和解码器的层数可以不同
        """

        """
        按照原始论文，整个编码器的输出本来是不需要再次归一化的，但是我们之前在 ResidualConnection 中修改了残差连接和归一化的顺序
        self.norm(x + self.dropout(sublayer(x))) vs x + self.dropout(sublayer(self.norm(x)))
        若采用前面原始论文的顺序，那么在编码器中，最后一个编码器块的输出就不需要再次归一化了，因为已经在 ResidualConnection 中进行了归一化。
        若采用后面的顺序，那么在编码器中，最后一个编码器块的输出需要再次归一化。
        """
        return self.norm(x)
