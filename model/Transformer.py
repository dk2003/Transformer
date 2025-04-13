import torch
import torch.nn as nn
from .components import *


class EncoderDecoder(nn.Module):
    """
    编码器-解码器结构，包含编码器和解码器
    """

    def __init__(
        self,
        # 编码器和解码器
        encoder: Encoder,
        decoder: Decoder,
        # 机器翻译任务中，有源和目标两种语言，因此需要两种不同的嵌入层
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        # 实际上，源位置编码和目标位置编码是共享的
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        # 投影层，将解码器的输出映射到目标语言的词表上，并且进行 log_softmax 操作
        projection_layer: ProjectionLayer,
    ) -> None:
        """
        :param encoder: 编码器
        :param decoder: 解码器
        :param src_embed: 源语言的嵌入层
        :param tgt_embed: 目标语言的嵌入层
        :param src_pos: 源语言的位置编码
        :param tgt_pos: 目标语言的位置编码
        :param projection_layer: 投影层，将解码器的输出映射到目标语言的词表上, 未进行 softmax 操作
        """
        super().__init__()
        self.encoder = encoder
        self.decoder: Decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

        # 初始化权重
        self._init_weights()

    def encode(self, src, src_mask):
        """
        编码器前向推理
        :param src: 源语言的输入序列，形状为 (batch_size, src_seq_len)
        :param src_mask: 源语言的掩码矩阵，形状为 (batch_size, 1, 1, src_seq_len)，用于屏蔽 <pad> 位置的信息
        :return: 编码器的输出，形状为 (batch_size, src_seq_len, d_model)
        """
        src_embed = self.src_embed(src)
        src_embed_pos = self.src_pos(src_embed)

        # encoder 编码器前向推理时接受两个参数：注入了位置线性的嵌入向量和掩码矩阵
        return self.encoder(src_embed_pos, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        解码器前向推理
        :param encoder_output: 编码器的输出，形状为 (batch_size, src_seq_len, d_model)
        :param src_mask: 源语言的掩码矩阵，形状为 (batch_size, 1, 1, src_seq_len)，用于屏蔽 <pad> 位置的信息
        :param tgt: 目标语言的输入序列，形状为 (batch_size, tgt_seq_len)
        :param tgt_mask: 目标语言的掩码矩阵，形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)，用于屏蔽 <pad> 位置的信息
        :return: 解码器的输出，形状为 (batch_size, tgt_seq_len, d_model)
        """
        tgt_embed = self.tgt_embed(tgt)
        tgt_embed_pos = self.tgt_pos(tgt_embed)
        return self.decoder(tgt_embed_pos, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码器前向传播, (Batch_size, seq_len, d_model)
        encoder_output = self.encode(src, src_mask)
        # 解码器前向传播,   (Batch_size, seq_len, d_model)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # 投影层：(Batch_size, seq_len, vocab_size)
        return self.project(decoder_output)

    def _init_weights(self):
        for p in self.parameters():  # 遍历模型所有可训练参数
            if p.dim() > 1:  # 只对维度大于 1 的参数（通常是矩阵）进行初始化
                nn.init.xavier_uniform_(p)  # 使用 Xavier 均匀分布初始化
        print("😜 初始化权重完成！")


class Transformer(EncoderDecoder):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ):
        # 编码器、解码器嵌入层
        src_embed = InputEmbeddings(src_vocab_size, d_model)
        tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)
        # 编码器、解码器位置编码
        src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
        tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

        # 创建编码器
        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(N)
            ]
        )
        encoder = Encoder(d_model, encoder_blocks)

        # 创建解码器
        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout,
                )
                for _ in range(N)
            ]
        )
        decoder = Decoder(d_model, decoder_blocks)

        # 创建投影层
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # 初始化父类
        super().__init__(
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            src_pos,
            tgt_pos,
            projection_layer,
        )
