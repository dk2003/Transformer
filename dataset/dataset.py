import torch
import torch.nn as nn
from torch.utils.data import Dataset


# 因果掩码
def causal_mask(size):
    """
    创建一个上三角矩阵，对角线以上的元素为0，对角线及其以下的元素为1（必须要将输入 ShiftRight 才能防止当前位置的信息泄露）
    返回值：shape=(1, size, size),例如如下是 size=4 的情况：
    tensor([[[1, 0, 0, 0],  # 第1个位置只能看自己，实际上是 <SOS>
            [1, 1, 0, 0],  # 第2个位置能看到前2个
            [1, 1, 1, 0],  # 第3个位置能看到前3个
            [1, 1, 1, 1]]]) # 第4个位置能看到全部4个
    """

    # torch.triu：获取矩阵的上三角部分（diagonal=0 包含对角线，diagonal=1 则不包含对角线），并将其他位置置 0。
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


# 构建双语数据集
class BilingualDataset(Dataset):
    def __init__(
        self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 句子开始的标记(start of sentense), 从 Number 转化为标量 tensor，这里直接使用的 tokenizer_src 获取 id，视频说和 tokenizer_tgt 获取的 id 是相同的
        # 通过查看 tokenizer 文件夹下的分词表，也可以发现特殊标记的 id 是相同的
        # 长整型 tensor，因为词汇表的索引可能超过 32 位长
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        # 句子结束(end of sentense)的标记
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        # 句子填充标记
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从原始数据集中获取未经处理的源语言和目标语言的句子
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # 将原始句子拆分为单个单词（token），然后根据 tokens 从词汇表中获取 token_ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        # 返回一个原始句子对应的 id 数组
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 我们是先将句子索引化，然后再添加特殊标记的索引。另外一种思路是先添加特殊标记，然后统一索引化
        # 之所以选择前者（更加稳健），可以防止某些 tokenizer 对特殊标记的分词可能不一致

        # 计算输入编码器需要填充的长度，2 表示为编码器的输入添加的一个开始标记和一个结束标记
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # 计算输入解码器需要填充的长度，1 表示为解码器的输入添加的一个开始标记，不需要添加结束标记
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # 如果输入的长度超过 seq_len，则抛出错误
            raise ValueError(f"Sentence is too long for sequence length {self.seq_len}")

        # 编码器嵌入层的输入，cat 用于连接多个张量
        encoder_input = torch.cat(
            [
                self.sos_token,  # 句子开始标记
                torch.tensor(
                    enc_input_tokens, dtype=torch.int64
                ),  # 源语言句子的 id 数组
                self.eos_token,  # 句子结束标记
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),  # 填充标记
            ],
            dim=0,
        )

        # 解码器嵌入层的输入，只需要添加 <SOS> 和 <PAD> ，不需要 <EOS>
        decoder_input = torch.cat(
            [
                self.sos_token,  # 句子开始标记
                torch.tensor(
                    dec_input_tokens, dtype=torch.int64
                ),  # 目标语言句子的 id 数组
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),  # 填充标记
            ],
            dim=0,
        )

        # 构建真实 label 标签，不需要 <SOS> 开始标记，需要 <EOS> 结束标记和 <PAD> 填充标记
        # 对比于解码器的输入，这里主要有以下两个区别：
        # 1. 解码器的输入中没有 <EOS> 结束标记，而真实标签中需要 <EOS> 结束标记，这是为了让模型自动预测句子的结束位置。
        # 2. 解码器的输入中含有 <SOS> 开始标记，用于将目标序列右移一位，保证预测当前位置时无法获取当前位置的真实词 token 信息
        label = torch.cat(
            [
                torch.tensor(
                    dec_input_tokens, dtype=torch.int64
                ),  # 目标语言句子的 id 数组
                self.eos_token,  # 句子结束标记
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),  # 填充标记
            ]
        )

        # 再次检查输入和标签的长度是否符合预期，确保它们的长度等于 seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # 编码器的输入, (seq_len)
            "decoder_input": decoder_input,  # 解码器的输入, (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # 编码器的掩码, (1, 1, seq_len) ，防止注意力提取 <pad> 信息
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()  # 解码器的第一层掩码, (1, 1, seq_len) ，防止注意力提取 <pad> 信息
            & causal_mask(
                decoder_input.size(0)
            ),  # 第二层因果掩码，防止解码器关注未来的信息， (1, 1, seq_len) & (1, seq_len, seq_len)，利用广播机制，将 (1, 1, seq_len) 扩展为 (1, seq_len, seq_len)
            "label": label,  # 真实标签, (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
