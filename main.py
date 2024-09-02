import os
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
    evaluate,
)

from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.MyDataset import MyDataSet
from torch.utils.data import DataLoader

from model import baseModel, BertTextModel_last_layer

from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter


'''设置 huggingface 的镜像'''
# import os
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

args = parse_args()
setup_seed(args)
setup_device(args)

start_time = time.time()
print("##############################################################")
print("                     Loading data...")
print("##############################################################")
print()
train_data = pd.read_csv(args.train_data_path)
dev_data = pd.read_csv(args.dev_data_path)



"""将pretrained model 下载下来，不然每次联网下载大概要一分半, 第二次运行并没有重新下载"""
'''这是基础的 base_bert '''
tokenizer = AutoTokenizer.from_pretrained(args.bert_pred)

train_dataset = MyDataSet(train_data, args=args, tokenizer=tokenizer, mode="train")
# train_dataset[0]

dev_dataset = MyDataSet(dev_data, args=args, tokenizer=tokenizer, mode="train")
# dev_dataset[0]

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

time_dif = get_time_dif(start_time)
print("##############################################################")

print("                 Loading data Time usage:", time_dif)

print("##############################################################")
print()


start_time = time.time()
print("##############################################################")
print("                     Inint model...")
print("##############################################################")
print()
# 实例化模型并将其移动到device设备上

'''(a) baseModel()'''
root, name = os.path.split(args.save_model_best)
args.save_model_best = os.path.join(root, str(args.select_model_last) + "_" +name)
root, name = os.path.split(args.save_model_last)
args.save_model_last = os.path.join(root, str(args.select_model_last) + "_" +name)

# 选择模型
if args.select_model_last:
    # 模型1
    model = baseModel(args=args).to(args.device)
else:
    # 模型2
    pass


'''(b) bert_TexCNN()'''
# root, name = os.path.split(args.save_model_best)
# args.save_model_best = os.path.join(root, str(args.select_model_last) + "_" +name)
# root, name = os.path.split(args.save_model_last)
# args.save_model_last = os.path.join(root, str(args.select_model_last) + "_" +name)

# # 选择模型
# if args.select_model_last:
#     # 模型1
#     model = BertTextModel_last_layer(args).to(args.device)
# else:
#     # 模型2
#     pass
#     # model = BertTextModel_encode_layer().to(args.device)

time_dif = get_time_dif(start_time)
print("##############################################################")

print("                 Inint model Time usage:", time_dif)

print("##############################################################")
print()
print(model.parameters)



num_total_steps = args.num_epochs * len(train_dataloader)
optimizer, scheduler = get_optimizer_and_scheduler(args, model, num_total_steps)

epochs = args.num_epochs

# 训练步骤

# 二分类交叉熵损失函数
# loss_fn = nn.BCELoss()


loss_sum = 0
total_step = 0

dev_best_loss = float("inf")
dev_best_F1_score = 0
last_improve = 0  # 记录上次验证集loss下降的batch数
flag = False  # 记录是否很久没有效果提升
writer = SummaryWriter(
    log_dir=args.log_path + "/" + time.strftime("%m-%d_%H.%M", time.localtime())
)


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

        # loss = loss_fn(out, label.to(torch.float32))
        loss = F.cross_entropy(out, label)
        loss_sum += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()

        """这里将来可以使用	tensorboard"""
        if total_step % 50 == 0:
            print('#########################################################')
            print(f"    train_data 50 个batch 的平均损失: {loss_sum/50}")
            loss_sum = 0
        '''修改为 1 用于测试'''
        if total_step % 10 == 1:
            # 每多少轮输出在训练集和验证集上的效果
            # true 为当前 train batch 的 预测值
            true = label.data.cpu()
            '''这里将来可以单独设置分类阈值, 采用sigmoid()'''
            # predic = torch.where(out >= 0.5, torch.tensor(1), torch.tensor(0))
            # predic = predic.data.cpu()
            
            '''采用 softmax()'''
            _, predic = torch.max(out.data, 1)
            predic = predic.detach().cpu()
            
            train_acc = metrics.accuracy_score(true, predic)
            
            train_f1_score = metrics.f1_score(true, predic)
            
            dev_acc, dev_f1_score, dev_loss = evaluate(dev_dataloader, model, args)

            """可以设置两部分结构，可以分别用 dev_loss 和 dev_fi 来保存模型"""
            
            # if dev_loss < dev_best_loss:
            #     dev_best_loss = dev_loss
            if dev_best_F1_score < dev_f1_score:
                dev_best_F1_score = dev_f1_score
                torch.save(
                    model.state_dict(),
                    f'{args.save_model_best}',
                )
                improve = "*"
                last_improve = total_step
            else:
                improve = ""
            time_dif = get_time_dif(start_time)

            """ 到时候再设置 f1 的输出格式吧"""
            msg = "Iter: {0:>4},|   Train Loss: {1:>4.2},|  Train Acc: {2:>6.2%},|  Train F1_score: {3:>6.2%}\
                Val Loss: {4:>5.2},|     Val Acc: {5:>6.2%},|   Val F1_score: {6:>6.2%},|   Time: {7} {8}"
            print('------------------------------------------------------------')
            print(
                msg.format(
                    total_step,
                    loss.item(),
                    train_acc,
                    train_f1_score,
                    dev_loss,
                    dev_acc,
                    dev_f1_score,
                    time_dif,
                    improve,
                )
            )
    

            """train 的 acc 和 loss 只是特定一个 batch 的，而 dev 的 loss 和 acc 是全部验证数据平均一个 batch 的值"""
            writer.add_scalar("loss/train", loss.item(), total_step)
            writer.add_scalar("loss/dev", dev_loss, total_step)
            writer.add_scalar("acc/train", train_acc, total_step)
            writer.add_scalar("acc/dev", dev_acc, total_step)
            writer.add_scalar("f1_score/train", train_f1_score, total_step)
            writer.add_scalar("f1_score/dev", dev_f1_score, total_step)
            model.train()

        if total_step - last_improve > args.require_improvement:
            # 验证集loss超过1000batch没下降，结束训练
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
        if flag == True:
            break

torch.save(model.state_dict(), f'{args.save_model_last}')
a = 1

