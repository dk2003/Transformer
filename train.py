"""
本模块是Transformer模型的训练脚本，用于加载数据集、构建模型、训练模型以及进行模型验证。
它包含了数据处理、模型构建、训练循环、解码和验证等功能。
"""

import warnings
from pathlib import Path
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import BilingualDataset, causal_mask

from model import Transformer

from config.config import get_config, get_weights_file_path
from tqdm import tqdm


def get_all_sentences(dataset, lang):
    """
    获取数据集的所有句子
    :param dataset: 数据集
    :param lang: 语言
    :return: 所有句子
    """
    for item in dataset:
        # Python 生成器函数，它会在每次迭代时返回一个值。
        yield item["translation"][lang]


# 不知道这种 tokenizer 的方式是否适合于中文？
def get_or_build_tokenizer(config, dataset, lang):
    # config["tokenizer_file"] = '../tokenizers/tokenizer_{lang}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang=lang))
    # 检查是否存在 tokenizer 文件
    if not Path.exists(tokenizer_path):  # 如果不存在，创建并训练 tokenizer
        # 1. parents=True：表示如果父目录不存在，会自动创建所有必要的父目录； 2. exist_ok=True：表示如果目录已经存在，不会抛出异常
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        """
        1. Tokenizer() : 创建HuggingFace tokenizers库的基础tokenizer对象
        2. WordLevel() : 指定使用词级别的 tokenizer 模型，它会将每个单词映射到唯一 ID
        3. unk_token="[UNK]" : 设置未知单词 OOV(未登录词)的标记为"[UNK]"，例如预测时，如果遇到训练时未见的单词，就会使用 [UNK] 标记。
        这相当于创建了一个基础的单词分割器，后续需要配合训练器(如代码中的 WordLevelTrainer)在数据集上训练，才能获得完整的 tokenizer 功能。
        """
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # 设置预分词器为空格分割模式 ，在正式 tokenize 之前，先按空白字符（空格、换行、制表符等）进行分割，例如将句子 "Hello world!" 预处理为 ["Hello", "world!"]
        # 保留标点符号与相邻单词的粘连（如 "world!" 保持整体），而不是将其分割成 "world" 和 "!" 。这是英文等空格分隔语言最基础的预处理方式
        tokenizer.pre_tokenizer = Whitespace()
        # 创建了一个 WordLevelTrainer 训练器对象，用于训练词级别的tokenizer。
        trainer = WordLevelTrainer(
            # 设置特殊标记（在词汇表中为这些特殊标记占住几个坑），这些特殊 token 会被优先分配固定的 ID（通常是前几个 ID），例如 UNK 通常为 0，PAD 为 1 等，方便快速查找
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            # 只有出现次数 ≥2 的单词才会被加入词汇表，出现 1 次的单词会被视为OOV(未登录词)，用 [UNK] 表示
            min_frequency=2,
        )
        # 使用生成器逐句提供训练数据，应用预分词器（Whitespace）进行初步分割, 统计所有单词的出现频率,根据 min_frequency 阈值筛选有效词汇, 自动分配每个词的唯一ID, 并构建词汇表。
        # 对于分词结果还会处理一些特殊情况例如：大小写变体（如"Hello"和"hello"），带标点的词（如"world!"和"world"），数字和特殊符号等。
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        # 将训练好的 tokenizer 序列化为JSON 文件
        tokenizer.save(str(tokenizer_path))
    else:
        # 如果存在，直接加载 tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# 加载数据集，获取 dataloader
def get_dataset(config):
    # 第一个参数：opus_books 数据集，第二个参数：数据集子集，第三个参数：加载训练集（opus-100 只提供了训练集，需要手动划分验证和测试集）
    dataset_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # 为源语言构建 tokenizer
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    # 为目标语言构建 tokenizer
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    # 训练集大小
    train_dataset_size = int(0.9 * len(dataset_raw))
    # 验证集大小
    val_dataset_size = len(dataset_raw) - train_dataset_size
    # 随机划分训练集和验证集，random_split() 是 PyTorch 提供的一个数据集划分工具函数，用于将数据集随机分割成多个子集。
    train_dataset_raw, val_dataset_raw = random_split(
        dataset_raw, [train_dataset_size, val_dataset_size]
    )

    # 创建训练数据集
    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # 创建验证数据集
    valid_dataset = BilingualDataset(
        val_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0  # 源语言句子的最大长度
    max_len_tgt = 0  # 目标语言句子的最大长度
    # 这些代码的作用是遍历整个数据集，统计源语言和目标语言的最大原始序列长度 （不含[SOS]/[EOS]/[PAD]等特殊标记）
    for item in dataset_raw:
        # 对源语言句子进行编码，获取token id列表
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        # 对目标语言句子进行编码，获取token id列表
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        # 更新源语言最大序列长度（不含特殊标记）
        max_len_src = max(max_len_src, len(src_ids))
        # 更新目标语言最大序列长度（不含特殊标记）
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(
        f"Max length of source sentence: {max_len_src}",
        f"Max length of target sentence: {max_len_tgt}",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,  # 将验证集的批量大小设置为 1，因为我们需要逐个样本进行验证
        shuffle=True,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len) -> Transformer:
    # 构建 Transformer 模型
    return Transformer(
        # 源语言的词汇表大小
        src_vocab_size=vocab_src_len,
        # 目标语言的词汇表大小
        tgt_vocab_size=vocab_tgt_len,
        # 源语言的序列长度（即输入句子的最大长度），这里使用了 config 中的 seq_len 参数
        src_seq_len=config["seq_len"],
        # 目标语言的序列长度（即输出句子的最大长度），这里使用了 config 中的 seq_len 参数
        tgt_seq_len=config["seq_len"],
        # 模型的嵌入维度，默认为 512，这里使用了 config 中的 d_model 参数
        d_model=config["d_model"],
        # 其余参数都使用默认值，如果模型难以训练，建议降低注意力头数 h 或者减少 Transformer 块的数量 N，但是这可能会影响模型的性能
        N=config["N"],  # Transformer 块的数量，默认为 6
        h=config["h"],  # 注意力头数，默认为 8
        dropout=config["dropout"],  # Dropout 概率，默认为 0.1
        d_ff=config["d_ff"],  # 前馈神经网络的隐藏层维度，默认为 2048
    )


def train_model(config):
    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 确保已经创建了 weights 文件夹
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # 加载数据集，获取 dataloader
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # 加载模型，并且转移到指定设备
    model: Transformer = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)

    # 构建 Tensorboard，会自动构建指定路径
    writer = SummaryWriter(config["experiment_name"])

    # 构建 Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # 恢复模型训练
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1  # 从下一个 epoch 开始训练
        optimizer.load_state_dict(state["optimizer_state_dict"])  # 加载优化器状态
        global_step = state["global_step"]  # 加载全局步数
        model.load_state_dict(state["model_state_dict"])  # 加载模型状态
        print(f"模型：{model_filename}恢复成功，从 epoch {initial_epoch} 开始训练")

    # 定义损失函数，为什么是交叉熵而不是 NLLLoss？
    # ignore_index: 对每个预测位置，先检查对应真实标签是否等于 ignore_index，如果等于，则该位置的损失计算会被跳过，如果不等于，正常计算交叉熵损失。这样可以防止模型学习无意义的填充标记
    # label_smoothing: 为标签应用标签平滑，防止模型过拟合
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        # 训练模型
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()  # 启用训练模式
            encoder_input = batch["encoder_input"].to(
                device
            )  # 编码器输入：[batch_size, seq_len]
            decoder_input = batch["decoder_input"].to(
                device
            )  # 解码器输入：[batch_size, seq_len]
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # 编码器掩码：[batch_size, h=1, 1, seq_len]
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # 解码器掩码：[batch_size, h=1, seq_len, seq_len]

            # 实际上我们已经定义了 forward 方法囊括了编码，解码和投影三个步骤,output.size=(Batch_size, seq_len, vocab_size))
            # output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # 获取真实标签，（Batch_size, seq_len）
            labels = batch["label"].to(device)

            """
            在 cv 中，当输入是4D张量 (N, C, H, W) 且目标是3D张量 (N, H, W) 时：
            1. 交叉熵损失会自动将输入视为 (N×H×W, C) 的形状
            2. 同时将目标视为 (N×H×W) 的形状
            3. 最终计算的是小批量中所有样本的所有像素的平均损失

            在 NLP 中，当输入是3D张量 (N, seq_len, vocab_size) 且目标是2D张量 (N, seq_len) 时：
            需要手动将输入和目标的形状转换为 (N×seq_len, vocab_size) 和 (N×seq_len)，然后才能正确计算损失。
            这是因为序列任务的输出形式多样（如有的模型只输出最后时间步），需要更灵活的手动控制
            """
            # output:(Batch_size, seq_len, vocab_size)——>(Batch_size * seq_len, vocab_size)
            # labels:(Batch_size, seq_len)——>(Batch_size * seq_len)
            loss = criterion(
                output.view(-1, tokenizer_tgt.get_vocab_size()), labels.view(-1)
            )
            # 在进度条上展示当前 Batch 的损失
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # 记录损失到 Tensorboard
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()  # 刷新缓冲区，确保数据被写入磁盘

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零

            global_step += 1  # 全局步数（批量数）加 1

            # 每个 epoch 都验证一次，以便我们快速观察效果
            if global_step % 200 == 0:
                run_validation(
                    model,
                    val_dataloader,
                    tokenizer_src,
                    tokenizer_tgt,
                    config["seq_len"],
                    device,
                    batch_iterator.write,
                    global_step,
                    writer,
                )

        # 每个 epoch 都保存模型
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                # 这里没有设置学习率调度器，Adam 优化器通过自适应地为每个权重分配学习率，避免了单独使用学习率调度器的需求，从而提高了模型性能并减少了参数管理的复杂性。
                # 在大多数情况下，Adam 的自适应性足够应对学习率调整。在某些情况下，余弦退火热重启策略等其他学习率调度方法也可能优于 Adam 优化器。因此，建议根据实验结果选择最适合当前任务的技术组合。
            },
            model_filename,
        )


def greedy_decode(
    model: Transformer,
    source,
    source_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
):
    """
    贪婪解码: 只进行一次编码，并且将其重用于解码器的所有解码器块和每个要预测的 token 上
    参数:
    - model: 训练好的Transformer模型实例
    - source: 源语言输入序列 (shape: [batch_size, seq_len])
    - source_mask: 源语言序列的掩码 (shape: [batch_size, 1, 1, seq_len])
    - tokenizer_src: 源语言tokenizer
    - tokenizer_tgt: 目标语言tokenizer
    - max_len: 生成序列的最大长度
    - device: 计算设备 (如 'cuda' 或 'cpu')
    返回值：
    - decoded_tokens: 生成的目标语言序列 (shape: [seq_len])
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")  # 获取 SOS 标记的 ID
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")  # 获取 EOS 标记的 ID

    # 计算出编码器的输出并且重用
    encoder_output = model.encode(source, source_mask)

    # 解码器的初始输入，即 <SOS> 标记
    # 创建一个形状为 [1,1] 的空张量（batch_size=1, seq_len=1），用目标语言的开始标记 [SOS] 的 ID 填充这个张量
    # type_as 确保张量类型与源语言输入张量相同（如float32/float16），to 将张量移动到指定设备（CPU/GPU）
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # 接下来，不断要求解码器输出下一个 token，直到达到最大长度或者输出了 EOS 标记
    while True:
        # decoder_input.size(1) 表示当前解码器已经输出的长度
        if decoder_input.size(1) == max_len:  # 如果已经达到最大长度，则停止生成
            break

        # 我们要明白一个事实，Transformer 的编、解码器其实没有限制输入的长度，我们之前规定为 seq_len 是为了方便批量训练
        # 为解码器的输入构建因果掩码
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # 计算解码器的输出,out.shape=(Batch_size=1,seq_len,d_model)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 根据解码器的输出获取最后一个 token 的投影，prob.shape=(Batch_size=1,vocab_size)
        prob = model.project(out[:, -1])
        # 在最后一个 token 的投影中选择概率最大的 id（这就是贪婪解码的思想，只考虑局部最优而不考虑全局最优），其实我们还可以使用束搜索
        # 返回值是一个包含两个元素的元组： (values, indices), next_word.shape=(batch_size=1,)
        _, next_word = torch.max(prob, dim=1)
        # decoder_input.shape=(Batch_size=1,seq_len)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )
        if next_word.item() == eos_idx:  # 如果输出了 EOS 标记，则停止生成
            break

    # # 返回解码器输出的序列，decoder_input.shape=(Batch_size=1,seq_len_predict)，以下返回值移除了批量维度
    return decoder_input.squeeze(0)


def run_validation(
    model,
    valid_data,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_state,
    writer,
    num_examples=2,
):
    """
    模型验证，由于在训练时，我们是可以看到整个预测序列的，因此我们可以直接将结合因果掩码进行多 token 的并行前向传播，还保证了未来的信息没有泄露
    但是模型验证也就是推理时
    """
    model.eval()  # 启用评估模式
    count = 0
    source_texts = []  # 源语言文本
    expected = []  # 期望输出(真实标签)
    predicted = []  # 预测输出

    console_width = 80
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗，加速推理
        for batch in valid_data:  # 验证集的 batch_size 设置为 1
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # 编码器输入
            encoder_mask = batch["encoder_mask"].to(device)  # 编码器掩码

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # 推理阶段，只计算一次编码器输出，并且将其重用于解码器的所有解码器块和每个要预测的 token 上
            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # 将模型输出转换为文本
            model_output_text = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy()
            )

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            # 控制台打印, 直接使用 print() 可能会影响 tqdm 的进度条，这里使用 tqdm 提供的打印方法
            print_msg("_" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_output_text}")
            if count == num_examples:
                break

    # 可以使用 torchMetrics 库来计算指标，并且记录在 tensorboard


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
