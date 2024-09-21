from argparse import ArgumentParser
import json


def save_args(args, args_name):
    root_path = args.args_folder
    args_save_path = f"{root_path}{args_name}"
    with open(args_save_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    return


def load_args(args, args_save_path):
    with open(args_save_path, "r") as f:
        args.__dict__ = json.load(f)
    return


if __name__ == "__main__":
    """暂时只测试了小样本"""
    parser = ArgumentParser()
    parser.add_argument("--data_count", type=str, default="small")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--resume", type=str, default="a/b/c.ckpt")
    parser.add_argument("--surgery", type=str, default="190")
    parser.add_argument(
        "--small_sample_args_folder", type=str, default="./small_sample/args/"
    )
    args = parser.parse_args()
    args_name = "test_args.txt"
    save_args(args, args_name)

    parser = ArgumentParser()
    args = parser.parse_args()
    args_save_path = "./small_sample/args/test_args.txt"
    load_args(args, args_save_path)

    print(args)
