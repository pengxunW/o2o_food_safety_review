import os
from argparse import ArgumentParser


"""!!! path 是相对于工作目录而言的，而非当前文件夹所在的路径"""


def mk_one_dir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(f"--- create new folder: path: {path} ---")
    else:
        print(f"--- {path} has exsist!  ---")


def mkdirs(args):
    print("创建项目所需的文件夹:")
    mk_one_dir(args.args_folder)  # 在模型超参数保存时会用到该路径
    mk_one_dir(args.log_folder)  # 在 Logger 类实例化时会用到该路径
    mk_one_dir(args.tensorboard_folder)  # 在 log_tensorboard() 会用到该路径
    mk_one_dir(args.model_save_folder)
    mk_one_dir(args.origin_result_folder)  # 生成保存结果的文件
    mk_one_dir(args.final_result_folder)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume", type=str, default="a/b/c.ckpt")
    parser.add_argument("--surgery", type=str, default="190")
    parser.add_argument("--args_save_folder", type=str, default="./others/test_args")
    args = parser.parse_args()
    mk_one_dir(args.args_save_folder)  # 调用函数
