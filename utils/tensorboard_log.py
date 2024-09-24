import time
import argparse
from torch.utils.tensorboard import SummaryWriter


def log_tensorboard(args, tw_name):
    root_path = args.tensorboard_folder
    tensorboard_file = f"{root_path}{tw_name}"
    tb_writer = SummaryWriter(log_dir=tensorboard_file)

    return tb_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用于测试 log_tensorboard 函数")

    parser.add_argument("--data_count", type=str, default="small", help="data count")
    parser.add_argument("--k_fold", type=int, default=2)
    parser.add_argument(
        "--small_sample_tensorboard_folder",
        type=str,
        default="./tensorboard/small_sample_tensorboard/",
    )
    parser.add_argument(
        "--main_tensorboard_folder", type=str, default="./tensorboard/main_tensorboard/"
    )
    args = parser.parse_args()
    # 小样本测试
    args.data_count = "small"
    for i in range(args.k_fold):
        tb_writer = log_tensorboard(args, i)
    # 大样本测试
    args.data_count = "big"
    for i in range(args.k_fold):
        tb_writer = log_tensorboard(args, i)
