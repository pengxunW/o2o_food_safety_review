from sklearn.model_selection import KFold
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from model import baseModel
from utils.MyDataset import MyDataSet
from torch.utils.data import DataLoader
from utils.tools import (
    setup_seed,
    setup_device,
    get_optimizer_and_scheduler,
    get_time_dif,
    evaluate,
)
from transformers import AutoTokenizer
from config import parse_args


# 交叉验证训练和测试模型


#########训练函数##########
def train(model, train_dataloader, dev_dataloader, args):

    num_total_steps = args.num_epochs * len(train_dataloader)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)
    epochs = args.num_epochs

    loss_sum = 0
    total_step = 0
    loss_fn = nn.CrossEntropyLoss()
    dev_best_loss = float("inf")
    dev_best_F1_score = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数

    """提前终止这里逻辑是有问题的"""
    flag = False  # 记录是否很久没有效果提升
    """这里将来可以使用 args 来代替"""
    epochs = 10
    for epoch in tqdm(range(epochs)):

        print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
        start_time = time.time()
        model.train()

        for batch in train_dataloader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["label"]
            total_step += 1
            out = model(input_ids, attention_mask)
            model.zero_grad()

            loss = loss_fn(out, label)
            loss_sum += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

            evaluate()
        # predict()

        """清除模型后"""
        return


def train_epoch(model, device, dataloader, loss_fn, optimizer):

    train_loss, train_correct = 0.0, 0
    model.train()
    for batch in dataloader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        out = model(input_ids, attention_mask)
        model.zero_grad()

        loss = loss_fn(out, labels)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()

        _, predictions = torch.max(out.data, 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss, train_correct


def valid_epoch(model, dataloader, loss_fn):
    """缺少 F1_score"""
    valid_loss, val_correct = 0.0, 0
    model.eval()
    # loss_fn = nn.BCELoss()
    # predict_all = np.array([], dtype=int)
    # labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            out = model(input_ids, attention_mask)
            # loss = loss_fn(out, label.to(torch.float32))
            loss = criterion(out, labels)
            valid_loss += loss

            labels = labels.data.cpu().numpy()

            """(a)这里使用 sigmoid() 后预测"""
            # """临界值这里将来也可以在 args 中设置一个参数"""
            # predic = torch.where(out >= 0.5, torch.tensor(1), torch.tensor(0))
            # """detach 这里是什么意思, detach()阻断反向传播，返回值仍为tensor, 在显存上"""
            # predic = predic.detach().cpu().numpy()

            """(b)采用 softmax()"""
            _, predictions = torch.max(out.data, 1)
            val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct

    #     acc = metrics.accuracy_score(labels_all, predict_all)
    #     f1_score = metrics.f1_score(labels_all, predict_all)
    # return acc, f1_score, loss_total / len(data_loader)


if __name__ == "__main__":
    
    args = parse_args()
    setup_seed(args)
    setup_device(args)

    """这里其实应该输入的是原始的完整的数据，为了简短，暂时使用 dev_data 代替"""
    train_data = pd.read_csv(args.train_data_path)
    dev_data = pd.read_csv(args.dev_data_path)

    data = dev_data

    criterion = nn.CrossEntropyLoss()
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pred)

    # 在第一个for循环中，从train_idx和val_idx中采样元素，然后将这些采
    # 样器转换为批大小等于128的DataLoader对象，初始化模型并将其传递给GPU，
    # 最后以0.002作为学习率来初始化Adam优化器。

    # 在第二个循环中，我们通过之前定义的函数训练和评估CNN模型，这些函数将
    # 返回所选训练集和测试集的损失和准确度。

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # 将所有的执行都保存到命名为 history 的字典里。在模型的训练和评估结束后，
    # 特定折叠（进入 history 字典）的所有分数都存储在字典中foldperf。
    foldperf = {}

    for i, (train_fold, test_fold) in enumerate(kf.split(data)):

        print("Fold: {}".format(i + 1))

        train_data = data.iloc[train_fold, :]
        dev_data = data.iloc[test_fold, :]
        train_dataset = MyDataSet(
            train_data, args=args, tokenizer=tokenizer, mode="train"
        )
        # train_dataset[0]
        """这里是把当做 test_data 还是 dev_data,mode 应该选择哪一个暂时还不清楚,还有下面的shuffle"""
        dev_dataset = MyDataSet(dev_data, args=args, tokenizer=tokenizer, mode="train")
        # dev_dataset[0]

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False
        )

        device = args.device
        model = baseModel(args=args).to(device)

        num_total_steps = args.num_epochs * len(train_dataloader)
        optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)
        num_epochs = args.num_epochs

        """这里将来可以添加 F1_score"""

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(num_epochs):

            train_loss, train_correct = train_epoch(
                model, device, train_dataloader, criterion, optimizer
            )
            test_loss, test_correct = valid_epoch(
                model, device, dev_dataloader, criterion
            )

            train_loss = train_loss / len(train_dataloader)
            train_acc = train_correct / len(train_dataloader) * 100
            test_loss = test_loss / len(dev_dataloader)
            test_acc = test_correct / len(dev_dataloader) * 100

            """这里输出格式将来要调整一下"""
            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1, num_epochs, train_loss, test_loss, train_acc, test_acc
                )
            )

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

        """这里可以选择最佳模型"""
        foldperf["fold{}".format(i + 1)] = history
        torch.save(model, "k_cross_CNN.pt")

        testl_f, tl_f, testa_f, ta_f = [], [], [], []
        """?????"""

        for f in range(1, k + 1):
            tl_f.append(np.mean(foldperf["fold{}".format(f)]["train_loss"]))
            testl_f.append(np.mean(foldperf["fold{}".format(f)]["test_loss"]))
            ta_f.append(np.mean(foldperf["fold{}".format(f)]["train_acc"]))
            testa_f.append(np.mean(foldperf["fold{}".format(f)]["test_acc"]))
            print("Performance of {} fold cross validation".format(k))
            print(
                "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc:  {:.2f}".format(
                    np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)
                )
            )

# if __name__ == "__main__":
#     print("Hello, World!")
