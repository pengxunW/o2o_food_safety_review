import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import time

from tqdm import tqdm
from argparse import ArgumentParser
from utils.save_and_load_args import load_args
from utils.tools import (
    get_time_dif,
)

from utils.MyDataset import MyDataSet
from torch.utils.data import DataLoader
from model import baseModel

if __name__ == "__main__":
    """可能需要修改的几个变量"""
    # args_save_path
    # test_data_path
    # model_save_folder, model_name
    # origin_result_path,

    parser = ArgumentParser()
    args = parser.parse_args()

    args_save_path = "./args/09-21_17-31/args.txt"
    load_args(args, args_save_path)

    k = args.k_fold

    print("---------------------------------------------------------")
    print("                   Loading sample data...")
    print("---------------------------------------------------------\n")
    start_time = time.time()

    test_data = pd.read_csv(args.test_data_path, sep=",")
    test_dataset = MyDataSet(test_data, args=args, mode="test")
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    time_dif = get_time_dif(start_time)
    print("----------------------------------------------------------")
    print("                 Loading data Time usage:", time_dif)
    print("----------------------------------------------------------\n")

    for i in range(k):

        print("Fold: {} : model_{} initial...".format(i, i))
        # 实例化模型并将其移动到device设备上
        """(a) baseModel()"""
        model = baseModel(args=args).to(args.device)
        model_name = f"model_{i}.pt"
        model_save_path = args.model_save_folder + model_name
        state_dict = torch.load(model_save_path)
        model.load_state_dict(state_dict)
        model.eval()

        print("model_{} start predicting...".format(i))

        all_probs = []
        all_scores = []
        all_preds = []

        """将来可以添加一个基于投票的结果获取方法"""
        with torch.no_grad():
            for batch in tqdm(test_dataloader):

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                out = model(input_ids, attention_mask)

                """(b)采用 softmax()"""
                out = F.softmax(out, dim=1)
                pre_probs = out.detach().cpu().numpy()
                all_probs.append(pre_probs)

                scores, pred_labels = torch.max(out.data, 1)
                pre_scores = scores.detach().cpu().numpy()
                pred_labels = pred_labels.detach().cpu().numpy()
                all_scores.append(pre_scores)
                all_preds.append(pred_labels)

        all_preds = np.concatenate(all_preds)
        all_scores = np.concatenate(all_scores)
        all_probs = np.concatenate(all_probs)

        columns = ["class_0", "class_1"]
        test_data[columns] = all_probs
        test_data["class_posibllity"] = all_scores
        test_data["predicted_class"] = all_preds
        predict_result_name = f"fold_{i}.csv"
        save_path = args.origin_result_folder + predict_result_name
        test_data.to_csv(save_path, index=None, sep=",")
    """将来可以加入更精细的时间进度提示"""
    a = 1
