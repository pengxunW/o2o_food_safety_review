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
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)


def get_optimizer_and_scheduler(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
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
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    """这里看不懂"""
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


def valid_epoch(valid_dataloader, model, loss_fn):

    valid_loss_in_epoch = []
    valid_loss_mean = 0.0
    valid_acc = 0.0
    valid_F1_score = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            out = model(input_ids, attention_mask)
            loss = loss_fn(out, labels)
            # 默认情况下，loss.item()就是一个batch内单个样本的平均loss
            valid_loss_in_epoch.append(loss.item())
            """(b)采用 softmax(),将预测结果的概率和转换为 1"""
            out = F.softmax(out,dim=1)
            scores, predictions = torch.max(out.data, 1)
            predictions = predictions.detach().cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predictions)

        valid_acc = metrics.accuracy_score(labels_all, predict_all)
        valid_F1_score = metrics.f1_score(labels_all, predict_all)
        valid_loss_mean = np.mean(valid_loss_in_epoch)

    evaluating_indicators = (valid_acc, valid_F1_score, valid_loss_mean)
    return evaluating_indicators


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
