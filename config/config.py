from pathlib import Path


def get_config():
    return {
        "batch_size": 12,
        "num_epochs": 50,
        "lr": 10
        ** -4,  # 在大多数情况下，Adam 的自适应性足够应对学习率调整，无需学习率调度器。在某些情况下，给定一个较大的学习率，配合余弦退火热重启策略等其他学习率调度方法也可能优于 Adam 优化器。因此，建议根据实验结果选择最适合当前任务的技术组合。
        "seq_len": 350,
        "d_model": 512,
        "N": 6,  # 6 层
        "h": 8,  # 8 个头
        "dropout": 0.1,  # 0.1 的 dropout 概率
        "d_ff": 2048,  # 2048 的 feedforward 层的维度
        "datasource": "opus_books",  # 数据集名称，opus_books 是一个包含英语和意大利语的数据集
        "lang_src": "en",  # 源语言
        "lang_tgt": "it",  # 目标语言
        "model_folder": "weights",  # 相对路径都是相对于脚本运行的位置
        "model_filename": "tmodel_",
        "preload": None,  # 重新启动的 epoch，例如在模型崩溃之后可以继续恢复训练
        "tokenizer_file": "tokenizer/tokenizer_{lang}.json",  # tokenizer 的保存路径
        "experiment_name": "runs/tmodel_1",  # tensorboard 的保存路径
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    # 在pathlib中，/ 运算符用于将路径组件连接起来。例如，Path("a") / "b" 会生成a/b的路径，而且Path 库具有跨平台特性，自动处理不同操作系统的路径分隔符（Windows用 \ ，Linux/macOS用 / ）
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    # glob 方法用于查找与指定模式匹配的文件路径。它返回一个生成器对象，可以通过迭代或使用 list() 函数将其转换为列表。
    model_glob = f"{model_basename}*.pt"
    # 找到所有匹配的文件路径
    model_paths = list(Path(model_folder).glob(model_glob))
    # 找到最新的文件路径，即最大的 epoch 数
    latest_model_path = max(model_paths, key=lambda p: int(p.stem.split("_")[-1]))
    return str(latest_model_path)


if __name__ == "__main__":
    print(get_weights_file_path(get_config(), "1"))  # weights\tmodel_1.pt
