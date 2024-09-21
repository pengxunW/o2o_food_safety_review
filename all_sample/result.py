import pandas as pd
from argparse import ArgumentParser
from utils.save_and_load_args import load_args


def predict_label(row):
    label = 0
    class_0 = row.iloc[0]
    class_1 = row.iloc[1]
    if class_0 > class_1:
        label = 0
    else:
        label = 1
    return label


def show_result_counts(result_path):

    result = pd.read_csv(result_path, sep=",")
    result["label"].value_counts()
    print(result["label"].value_counts())


if __name__ == "__main__":
    """可能需要修改的几个变量"""
    # args_save_path
    # test_data_path
    # model_save_folder, model_name
    # origin_result_folder, final_result_folder
    parser = ArgumentParser()
    args = parser.parse_args()
    args_save_path = "./args/09-21_11-21/args.txt"
    load_args(args, args_save_path)
    """权重这里也要修改"""
    k = args.k_fold
    weights = [0.5, 0.5]
    test_data_path = args.test_data_path
    origin_result_folder = args.origin_result_folder
    result_folser = args.final_result_folder
    result_save_path = result_folser + f"sample_result_{k}_fold.csv"

    """修改部分"""
    result = pd.read_csv(test_data_path, sep=",")
    columns = ["class_0", "class_1"]

    result[columns] = 0
    result["label"] = 0
    for i in range(k):
        file_path = origin_result_folder + f"fold_{i}.csv"
        fold_data = pd.read_csv(file_path, sep=",")
        result[columns] += fold_data[columns] * weights[i]

    result["label"] = result[columns].apply(
        predict_label, axis=1
    )  # 按行来 [class_0, class_1]
    result[["id", "label"]].to_csv(result_save_path, sep=",", index=False)

    show_result_counts(result_save_path)
