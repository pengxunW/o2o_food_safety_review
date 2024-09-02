import os
import numpy as np
import pandas as pd

import torch

import time

from tqdm import tqdm
from config import parse_args

from utils.tools import (
    setup_seed,
    setup_device,
    get_time_dif,
)

from transformers import AutoTokenizer

from utils.MyDataset import MyDataSet
from torch.utils.data import DataLoader

from model import baseModel, BertTextModel_last_layer

args = parse_args()
setup_seed(args)
setup_device(args)


print("##############################################################")
print("                   Loading data...")
print("##############################################################")
print()

start_time = time.time()
test_data = pd.read_csv(args.test_data_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

test_dataset = MyDataSet(test_data, args=args, tokenizer=tokenizer, mode="test")
test_dataset[0]

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


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

# 实例化模型并将其移动到device设备上
# model = baseModel(args=args).to(args.device)
# saved_dict/base_bert/model_0.3726568818092346.pt
# state_dict = torch.load('saved_dict/base_bert/model_0.3726568818092346.pt')
state_dict = torch.load(args.save_model_best)
model.load_state_dict(state_dict)
model.eval()

start_time = time.time()
print("##############################################################")

print("                     Starting Predict...")

print("##############################################################")
print()

predict_all = np.array([], dtype=int)
with torch.no_grad():
    for batch in tqdm(test_dataloader):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        out = model(input_ids, attention_mask)

        '''(a)这里使用 sigmoid() 后预测'''
            # """临界值这里将来也可以在 args 中设置一个参数"""
            # predic = torch.where(out >= 0.5, torch.tensor(1), torch.tensor(0))
            # """detach 这里是什么意思, detach()阻断反向传播，返回值仍为tensor, 在显存上"""
            # predic = predic.detach().cpu().numpy()
            
        '''(b)采用 softmax()'''
        _, predic = torch.max(out.data, 1)
        predic = predic.detach().cpu().numpy()
        predict_all = np.append(predict_all, predic)
    
test_data["label"] = predict_all
test_data.to_csv(args.result_save_path_with_comment, index=None, sep=",")
# test_data[["id", "label"]].to_csv(args.result_save_path, index=None, sep=",")

time_dif = get_time_dif(start_time)
print("##############################################################")

print("                 Predict Time usage:", time_dif)

print("##############################################################")
print()

test_data

a = 1
