import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from tqdm import tqdm
from config import parse_args

from utils.tools import (
    setup_seed,
    setup_device,
    get_optimizer_and_scheduler,
    get_time_dif,
    train_epoch,
    valid_epoch,
)

from utils.log import Logger
from utils.MyDataset import MyDataSet

from torch.utils.data import DataLoader

from model import baseModel
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args)
    setup_device(args)

    print("---------------------------------------------------------")
    print("                   Loading sample data...")
    print("---------------------------------------------------------\n")
    # 如果使用使用交叉验证，在划分数据时，valid_data 就不使用了，直接用 train_data 测试
    start_time = time.time()

    train_data = pd.read_csv(args.sample_train_data_path)
    valid_data = pd.read_csv(args.sample_valid_data_path)
    data = pd.concat([train_data, valid_data], axis=0).reset_index(drop=True)

    time_dif = get_time_dif(start_time)
    print("----------------------------------------------------------")
    print("                 Loading data Time usage:", time_dif)
    print("----------------------------------------------------------\n")

    print("----------------------------------------------------------")
    print("                     Inint model...")
    print("----------------------------------------------------------\n")
    start_time = time.time()

    # # 实例化模型并将其移动到device设备上
    # """(a) baseModel()"""
    # model = baseModel(args=args).to(args.device)
    # print(model.parameters)

    time_dif = get_time_dif(start_time)
    print("----------------------------------------------------------")
    print("                 Inint model Time usage:", time_dif)
    print("----------------------------------------------------------\n")

    # 在第一个for循环中，从train_idx和val_idx中采样元素，然后将这些采
    # 样器转换为批大小等于 batch_size 的 DataLoader 对象，初始化模型并将其传递给GPU，
    # 最后以0.002作为学习率来初始化Adam优化器。

    # 在第二个循环中，我们通过之前定义的函数训练和评估CNN模型，这些函数将
    # 返回所选训练集和测试集的损失和准确度。
    """将来可以将折数添加到配置文件中"""

    k = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 将所有的执行都保存到命名为 history 的字典里。在模型的训练和评估结束后，
    # 特定折叠（进入 history 字典）的所有分数都存储在字典中 evaluating_indicators_per_fold。
    """这个字典现在没用到，将来再保存损失"""
    evaluating_indicators_per_fold = {}

    logger = Logger()
    for i, (train_fold, test_fold) in enumerate(kf.split(data)):
        print("-----------------Fold: {} --------------------------".format(i))
        """很重要"""
        # 保证索引是连续的，否则之后 dataloader 中 __getitem__ 可能取数据范围之外的索引
        # 因为 dataloader 是依据传入数据的大小产生对应范围索引，比如数据大小是 100，那么所有索引就是[0:99]
        train_data = data.iloc[train_fold, :].reset_index(drop=True)
        valid_data = data.iloc[test_fold, :].reset_index(drop=True)

        train_dataset = MyDataSet(train_data, args=args, mode="train")
        valid_dataset = MyDataSet(valid_data, args=args, mode="train")

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False
        )

        # 实例化模型并将其移动到device设备上
        """(a) baseModel()"""
        model = baseModel(args=args).to(args.device)
        model_name = f"model_{i}.pt"
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
        tb_writer = SummaryWriter(
            log_dir=args.log_path + "/" + time.strftime("%m-%d_%H-%M", time.localtime())
        )
        # 写入日志
        logger.logger.info(f"\n------------------------------------------------")
        logger.logger.info(f"start training in {i} fold......\n")
        # 监测记录每折训练模型每轮对应的指标
        history = {"valid_loss_mean": [], "valid_acc": [], "valid_F1_score": []}
        for epoch in tqdm(range(epochs)):

            print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
            start_time = time.time()

            train_epoch(train_dataloader, model, loss_fn, optimizer, scheduler)

            """暂时只输出记录每折验证集每轮对应的损失，之后可以添加训练集的"""
            evaluating_indicators = valid_epoch(valid_dataloader, model, loss_fn)

            valid_acc = evaluating_indicators[0]
            valid_F1_score = evaluating_indicators[1]
            valid_loss_mean = evaluating_indicators[2]
            history["valid_loss_mean"].append(valid_loss_mean)
            history["valid_acc"].append(valid_acc)
            history["valid_F1_score"].append(valid_F1_score)
            evaluating_indicators_per_fold["fold_{}".format(i + 1)] = history

            """但现在模型结构还不完整，比如将来添加交叉验证之后，模型的路径和名称"""
            # 设置保存模型的逻辑
            if valid_best_F1_score < valid_F1_score:
                valid_best_F1_score = valid_F1_score
                torch.save(
                    model.state_dict(),
                    f"{args.model_save_path + model_name}",
                )
                improve = "*"
                last_improve = epoch
            else:
                improve = ""
            time_dif = get_time_dif(start_time)
            """将来可以将输出的相关指标封装成一个函数"""

            # tensorboard可视化
            tb_writer.add_scalar("Validation/valid_loss_mean", valid_loss_mean, epoch)
            tb_writer.add_scalar("Validation/valid_acc", valid_acc, epoch)
            tb_writer.add_scalar("valid_F1_score/Validation", valid_F1_score, epoch)
            # logger.logger.info('%d epoch train mean loss: %.2f \n'%(epoch,mean_loss))
            # logger.logger.info('%d epoch train mean acc: %.2f \n'%(epoch,mean_acc))

            msg_1 = "Epoch: {0:>3},     Time: {1} {2}"
            msg_2 = (
                "Val Loss: {0:>5.2},     Val Acc: {1:>6.2%},   Val F1_score: {2:>6.2%}\n"
            )
            logger.logger.info(msg_1.format(epoch, time_dif, improve))
            logger.logger.info(msg_2.format(valid_loss_mean, valid_acc, valid_F1_score))

            # 早停逻辑结构
            if epoch - last_improve > args.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

    with open("./log/evaluating_indicators_per_fold.pkl", "wb") as f:
        pickle.dump(evaluating_indicators_per_fold, f)
    # 从pkl文件中加载字典
    with open("./log/evaluating_indicators_per_fold.pkl", "rb") as f:
        loaded_dict = pickle.load(f)
    # 打印加载后的字典
    print(loaded_dict)
