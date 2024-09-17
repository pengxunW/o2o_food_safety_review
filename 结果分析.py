import os
import pandas as pd
from tqdm import tqdm
from config import parse_args

# args = parse_args()

# result_with_commnet_data_path = args.result_save_path_with_comment
# result_data_path = args.result_save_path

# '''原始预测结果的标签统计'''
# data1 = pd.read_csv(result_with_commnet_data_path, sep=',')
# data1['label'].value_counts()
# print(data1['label'].value_counts())

# '''数据增强后预测结果的标签统计'''
# data2 = pd.read_csv(result_data_path, sep=',')
# data2['label'].value_counts()
# print(data2['label'].value_counts())
# a = 1

args = parse_args()
result_roberta_path = "data/result/chinese-roberta-wwm-ext-large_result_0.928.csv"
TextCNN_path = "data/result/TextCnn_result_4.csv"

test_data = pd.read_csv(args.test_data_path)
result_roberta = pd.read_csv(result_roberta_path)
result_text_cnn = pd.read_csv(TextCNN_path)

data = pd.merge(test_data, result_roberta, how="left", on="id")
data = data.merge(result_text_cnn, how="left", on="id")
data.columns = ["id", "comment", "label_1", "label_2"]

key_index_file_path = "data/data_enhance/"
key_index_with_1 = "key_index_with_1.txt"
key_index_with_0 = "key_index_with_0.txt"

indexex_0 = []
indexex_1 = []
indexex = []
if not os.path.exists(key_index_file_path):
    print(f'{key_index_file_path} 路径不存在！') 
else:
    file_0 = key_index_file_path + key_index_with_0
    file_1 = key_index_file_path + key_index_with_1
    file = key_index_file_path + key_index_with_1
    with open(file, mode='r',encoding='utf-8') as f:
        for line in tqdm(f):
            index = int(line.strip())
            indexex.append(index)
    # with open(file_0, mode='r',encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         index = int(line.strip())
    #         indexex_0.append(index)

    # with open(file_1, mode='r',encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         index = int(line.strip())
    #         indexex_1.append(index)
    
indexex = list(set(indexex))
# indexex_0 = list(set(indexex_0))        
# indexex_1 = list(set(indexex_1))
df = data.iloc[indexex]
df.to_csv('key_data_2.csv', sep=',')
print(df)

print(df['label_1'].value_counts())
a = 1