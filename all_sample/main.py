"""将文件名改为 small_sample_main.py"""

"""求算模型平均指标的模块没完成"""
"""将来添加一个功能，创建项目的框架文件夹"""
"""将来为什么不把小样本所有数据放在同一个项目文件夹下呢"""
"""更新 gitignore"""

import time

# import pickle
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm
from config import parse_args

from utils.tools import (
    setup_seed,
    setup_device,
    get_optimizer_and_scheduler,
    get_time_dif,
    train_epoch,
    evaluate_epoch,
)
from utils.mkdir import mkdirs
from utils.save_and_load_args import save_args
from utils.tensorboard_log import log_tensorboard
from utils.log import Logger

from utils.MyDataset import MyDataSet

from torch.utils.data import DataLoader

from model import baseModel
from sklearn.model_selection import KFold

if __name__ == "__main__":
    # args.k_fold,  args.bert_pred
    args = parse_args()
    setup_seed(args)
    setup_device(args)
    """这里单独的显示，是为了后面使用 Dataloader 打乱时，固定打乱顺序，确保结果可复现"""
    args.seed = 2024
    args.k_fold = 2

    mkdirs(args)
    args.bert_pred = "./pretrained_models/uerroberta-base-finetuned-dianping-chinese"
    args.batch_size = 8
    args.num_epochs = 20
    args.warmup_ratio = 0.1
    args.learning_rate = 1e-3
    args.require_improvement = 1

    args.max_seq_len = 100
    # 保存超参数
    args_name = "args.txt"
    save_args(args, args_name)
    # 设置 log 的保存路径
    """将 log 的 path 更改一下"""
    log_name = "main.log"
    logger = Logger(args, log_name)
    print("---------------------------------------------------------")
    print("                   Loading all data...")
    print("---------------------------------------------------------\n")
    # 如果使用使用交叉验证，在划分数据时，valid_data 就不使用了，直接用 train_data 测试
    start_time = time.time()

    data = pd.read_csv(args.train_data_path, sep="\t")
    time_dif = get_time_dif(start_time)
    print("----------------------------------------------------------")
    print("                 Loading all data Time usage:", time_dif)
    print("----------------------------------------------------------\n")

    # 在第一个for循环中，从train_idx和val_idx中采样元素，然后将这些采
    # 样器转换为批大小等于 batch_size 的 DataLoader 对象，初始化模型并将其传递给GPU，
    # 最后以0.002作为学习率来初始化Adam优化器。

    # 在第二个循环中，我们通过之前定义的函数训练和评估CNN模型，这些函数将
    # 返回所选训练集和测试集的损失和准确度。

    k_fold = args.k_fold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # 将所有的执行都保存到命名为 history 的字典里。在模型的训练和评估结束后，
    # 特定折叠（进入 history 字典）的所有分数都存储在字典中 evaluating_indicators_per_fold。
    """这个字典现在没用到，将来再保存损失"""
    # evaluating_indicators_per_fold = {}

    for i, (train_fold, test_fold) in enumerate(kf.split(data)):
        print("--------------------Fold: {} ------------------------".format(i))
        """很重要"""
        # 保证索引是连续的，否则之后 dataloader 中 __getitem__ 可能取数据范围之外的索引
        # 因为 dataloader 是依据传入数据的大小产生对应范围索引，比如数据大小是 100，那么所有索引就是[0:99]

        train_data = data.iloc[train_fold, :].reset_index(drop=True)
        valid_data = data.iloc[test_fold, :].reset_index(drop=True)

        train_dataset = MyDataSet(train_data, args=args, mode="train")
        valid_dataset = MyDataSet(valid_data, args=args, mode="train")

        """打乱时是否需要固定随机数种子，全局随机数种子已在 setup_seed() 中设置过"""
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False
        )

        # 实例化模型并将其移动到device设备上
        """(a) baseModel()"""
        model = baseModel(args=args).to(args.device)
        model_name = f"model_{i}.pt"
        model_save_path = args.model_save_folder + model_name
        # print(model.parameters)

        loss_fn = nn.CrossEntropyLoss()
        num_total_steps = args.num_epochs * len(train_dataloader)
        optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)
        epochs = args.num_epochs

        # 训练步骤
        # 二分类交叉熵损失函数

        valid_best_F1_score = 0.0
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升

        # 每一折对应一个 tb_writer
        tw_name = f"fold_{i}"
        tb_writer = log_tensorboard(args, tw_name)
        # 写入日志
        logger.logger.info(f"一共 {k_fold} fold, 进度：[{i+1}/{k_fold}]\n")
        logger.logger.info(f"------------------------------------------------")
        logger.logger.info(f"start training in {i} fold......\n")
        # 监测记录每折训练模型每轮对应的指标
        # history = {"valid_loss_mean": [], "valid_acc": [], "valid_F1_score": []}
        for epoch in tqdm(range(epochs)):

            print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
            start_time = time.time()

            train_epoch(train_dataloader, model, loss_fn, optimizer, scheduler)

            """暂时只输出记录每折验证集每轮对应的损失，之后可以添加训练集的"""

            train_metrics = evaluate_epoch(train_dataloader, model, loss_fn)
            valid_metrics = evaluate_epoch(valid_dataloader, model, loss_fn)

            train_acc = train_metrics[0]
            train_F1_score = train_metrics[1]
            train_loss_mean = train_metrics[2]

            valid_acc = valid_metrics[0]
            valid_F1_score = valid_metrics[1]
            valid_loss_mean = valid_metrics[2]

            # history["valid_loss_mean"].append(valid_loss_mean)
            # history["valid_acc"].append(valid_acc)
            # history["valid_F1_score"].append(valid_F1_score)
            # evaluating_indicators_per_fold["fold_{}".format(i + 1)] = history

            """但现在模型结构还不完整，比如将来添加交叉验证之后，模型的路径和名称"""
            # 设置保存模型的逻辑
            if valid_best_F1_score < valid_F1_score:
                valid_best_F1_score = valid_F1_score
                torch.save(
                    model.state_dict(),
                    model_save_path,
                )
                improve = "*"
                last_improve = epoch
            else:
                improve = ""
            time_dif = get_time_dif(start_time)
            """将来可以将输出的相关指标封装成一个函数"""
            """标题有问题"""
            # tensorboard可视化
            tb_writer.add_scalar(f"train_fold_{i}_loss", train_loss_mean, epoch)
            tb_writer.add_scalar(f"train_fold_{i}_acc", train_acc, epoch)
            tb_writer.add_scalar(f"train_fold_{i}_F1_score", train_F1_score, epoch)

            tb_writer.add_scalar(f"valid_fold_{i}_loss", valid_loss_mean, epoch)
            tb_writer.add_scalar(f"valid_fold_{i}_acc", valid_acc, epoch)
            tb_writer.add_scalar(f"valid_fold_{i}_F1_score", valid_F1_score, epoch)

            head_msg = "Epoch: {0:>3},   Time: {1},  Improve: {2}"
            logger.logger.info(head_msg.format(epoch, time_dif, improve))

            train_msg = "Train Loss: {0:>5.2},     Train Acc: {1:>6.2%},   Train F1_score: {2:>6.2%}"
            logger.logger.info(
                train_msg.format(train_loss_mean, train_acc, train_F1_score)
            )

            valid_msg = "Val Loss: {0:>5.2},     Val Acc: {1:>6.2%},   Val F1_score: {2:>6.2%}\n"
            logger.logger.info(
                valid_msg.format(valid_loss_mean, valid_acc, valid_F1_score)
            )

            # 早停逻辑结构
            if epoch - last_improve > args.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

    # with open("./log/evaluating_indicators_per_fold.pkl", "wb") as f:
    #     pickle.dump(evaluating_indicators_per_fold, f)
    # # 从pkl文件中加载字典
    # with open("./log/evaluating_indicators_per_fold.pkl", "rb") as f:
    #     loaded_dict = pickle.load(f)
    # # 打印加载后的字典
    # print(loaded_dict)
