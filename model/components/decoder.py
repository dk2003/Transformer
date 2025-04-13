import torch
import torch.nn as nn
from .residual_connection import ResidualConnection
from .multihead_attention import MultiHeadAttentionBlock
from .feed_forward import FeedForwardBlock
from .layer_normalization import LayerNormalization


class DecoderBlock(nn.Module):
    """
    解码器块，包含自注意力层、交叉注意力层和前馈层
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        """
        :param features: 模型的特征维度
        :param self_attention_block: 自注意力层
        :param cross_attention_block: 交叉注意力层
        :param feed_forward_block: 前馈层
        :param dropout: dropout 概率
        """
        super(DecoderBlock, self).__init__()
        # 自注意力层，目标句子在观察目标句子
        self.self_attention_block = self_attention_block
        # 交叉注意力层，目标句子在观察源句子
        self.cross_attention_block = cross_attention_block
        # 前馈层
        self.feed_forward_block = feed_forward_block

        # 三个带有残差连接的子层
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: 注入了位置编码的目标序列或者是上一个解码器块的输出，形状为 (batch_size, seq_len, d_model)
        :param encoder_output: 编码器的输出，形状为 (batch_size, seq_len, d_model)
        :param src_mask: 源序列的 mask，掩盖 <pad> 的权重，形状为 (batch_size, 1, 1, seq_len)
        :param tgt_mask: 目标序列的 mask，掩盖 <pad> 和未来词的权重，形状为 (batch_size, 1, seq_len, seq_len)
        :return: 输出，形状为 (batch_size, seq_len, d_model)
        """
        # 自注意力层
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask),
        )
        # 交叉注意力层
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        # 前馈层
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    """
    解码器，由多个解码器块组成，每个解码器块包含自注意力层、交叉注意力层和前馈层
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        """
        :param features: 模型的特征维度
        :param layers: 多个解码器块组成的列表
        """
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        """
        :param x: 注入了位置编码的目标序列，形状为 (batch_size, seq_len, d_model)
        :param encoder_output: 编码器的输出，形状为 (batch_size, seq_len, d_model)
        :param src_mask: 源序列的 mask，掩盖 <pad> 的权重，形状为 (batch_size, 1, 1, seq_len)
        :param tgt_mask: 目标序列的 mask，掩盖 <pad> 和未来词的权重，形状为 (batch_size, 1, seq_len, seq_len)
        :return: 输出，形状为 (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)
