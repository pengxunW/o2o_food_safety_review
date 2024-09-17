import os
import pandas as pd
from config import parse_args

args = parse_args()
result_roberta_path = "data/result/chinese-roberta-wwm-ext-large_result_0.928.csv"
TextCNN_path = "data/result/TextCnn_result_4.csv"

test_data = pd.read_csv(args.test_data_path)
result_roberta = pd.read_csv(result_roberta_path)
result_text_cnn = pd.read_csv(TextCNN_path)

data = pd.merge(test_data, result_roberta, how="left", on="id")
data = data.merge(result_text_cnn, how="left", on="id")
data.columns = ["id", "comment", "label_1", "label_2"]
df = data[data.label_1 != data.label_2]

df.to_csv("key_data.csv", sep=",")
data = pd.read_csv('key_data.csv')

index_with_1 = df.index
index_with_0 = [981, 1017, 1886]

key_index_file_path = "data/data_enhance/"
key_index_with_1 = "key_index_with_1.txt"
key_index_with_0 = "key_index_with_0.txt"

if not os.path.exists(key_index_file_path):
    os.makedirs(key_index_file_path)
    print(f"{key_index_file_path} 文件路径创建成功！")
    file = key_index_file_path+key_index_with_1
    with open(file, mode="w", encoding="utf-8") as f:
        for i in index_with_1:
            if i not in index_with_0:
                f.write(str(i) + "\n")
else:
    file = key_index_file_path+key_index_with_1
    with open(file, mode="w", encoding="utf-8") as f:
        for i in index_with_1:
            if i not in index_with_0:
                f.write(str(i) + "\n")



if not os.path.exists(key_index_file_path):
    os.makedirs(key_index_file_path)
    print(f"{key_index_file_path} 文件路径创建成功！")
    file = key_index_file_path+key_index_with_0
    with open(file, mode="w", encoding="utf-8") as f:
        for i in index_with_0:
                f.write(str(i) + "\n")
else:
    file = key_index_file_path+key_index_with_0
    with open(file, mode="w", encoding="utf-8") as f:
        for i in index_with_0:
                f.write(str(i) + "\n")
