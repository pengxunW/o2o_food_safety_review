import os
import time
from datetime import timedelta
from sklearn import metrics
import torch
import torch.nn.functional as F
import random
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup


"""这里将来要将模型迁移到 gpu 设备上"""


def setup_device(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


"""种子这里可能还要专门核对一下，用于张量计算"""


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为当前GPU设置随机种子；
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


def get_optimizer_and_scheduler(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    # 将参数分为两类：需要 weight_decay 和不需要 weight_decay

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()  # 获取模型参数的名称和值
                if not any(
                    nd in n for nd in no_decay
                )  # 不包含 "bias" 和 "LayerNorm.weight"
            ],
            "weight_decay": args.weight_decay,  # 对这些参数应用 weight_decay
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # 优化器 (optimizer)
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    """这里看不懂"""
    # 学习率调度器 (scheduler)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_ratio * num_total_steps,
        num_training_steps=num_total_steps * 1.05,
    )
    return optimizer, scheduler


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train_epoch(train_dataloader, model, loss_fn, optimizer, scheduler):

    model.train()
    for batch in train_dataloader:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        out = model(input_ids, attention_mask)
        optimizer.zero_grad()
        model.zero_grad()
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return


"""该函数用于评估训练集和验证集一轮的损失"""


def evaluate_epoch(dataloader, model, loss_fn):

    loss_one_epoch = []
    loss_mean = 0.0
    acc = 0.0
    F1_score = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            out = model(input_ids, attention_mask)
            loss = loss_fn(out, labels)
            # 默认情况下，loss.item()就是一个batch内单个样本的平均loss
            loss_one_epoch.append(loss.item())
            """(b)采用 softmax(),将预测结果的概率和转换为 1"""

            out = F.softmax(out, dim=1)
            scores, pred_labels = torch.max(out.data, 1)
            pred_labels = pred_labels.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_labels.append(labels)
            all_preds.append(pred_labels)
        # 将 list [batch_0, batch_1,...batch_n] 整个拼接起来，形成一行
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        acc = metrics.accuracy_score(all_labels, all_preds)
        F1_score = metrics.f1_score(all_labels, all_preds)
        loss_mean = np.mean(loss_one_epoch)
    return (acc, F1_score, loss_mean)


"""将来这里的函数可以删除了"""


def evaluate(data_loader, model, args):
    model.eval()
    loss_total = 0
    # loss_fn = nn.BCELoss()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["label"]

            out = model(input_ids, attention_mask)
            # loss = loss_fn(out, label.to(torch.float32))
            loss = F.cross_entropy(out, label)
            loss_total += loss

            label = label.data.cpu().numpy()

            """(a)这里使用 sigmoid() 后预测"""
            # """临界值这里将来也可以在 args 中设置一个参数"""
            # predic = torch.where(out >= 0.5, torch.tensor(1), torch.tensor(0))
            # """detach 这里是什么意思, detach()阻断反向传播，返回值仍为tensor, 在显存上"""
            # predic = predic.detach().cpu().numpy()

            """(b)采用 softmax()"""
            _, predic = torch.max(out.data, 1)
            predic = predic.detach().cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all)
    return acc, f1_score, loss_total / len(data_loader)
