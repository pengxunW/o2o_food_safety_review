import time
import argparse
import logging  # 引入logging模块
from pathlib import Path

"""得到项目的根目录"""


def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)


class Logger:
    def __init__(self, args, log_name, mode="a"):
        root_path = args.log_folder

        # 第一步，创建一个logger
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        log_name = f"{root_path}{log_name}"
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)  # 日志文件追加在末尾
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


if __name__ == "__main__":
    """只测试了小样本数据"""
    parser = argparse.ArgumentParser(description="用于测试 Logger 类的参数设置")
    parser.add_argument("--data_count", type=str, default="small", help="data count")
    parser.add_argument(
        "--small_sample_log_folder", type=str, default="./small_sample/logs/"
    )
    # parser.add_argument("--main_log_folder", type=str, default="./log/main_log/")
    args = parser.parse_args()
    log_name = "test_logger.log"
    logger_small = Logger(args, log_name)
    # 小样本写入日志
    logger_small.logger.info(
        "这是一条测试 small_data 日志模块能否正常运行的记录......\n"
    )
    # args.data_count = "all"
    # logger_main = Logger(args)
    # # 大样本写入日志
    # logger_main.logger.info("这是一条测试 all_data 日志模块能否正常运行的记录......\n")
