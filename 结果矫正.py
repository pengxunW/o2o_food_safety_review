import os
import pandas as pd
from tqdm import tqdm
from config import parse_args

args = parse_args()

result_data_path = args.result_save_path

final_result_data_path = 'data/result/final_result.csv'


data = pd.read_csv('data/result/chinese-roberta-wwm-ext-large_result_0.928.csv', sep=',')


key_index_file_path = "data/data_enhance/"
key_index_with_1 = "key_index_with_1.txt"
key_index_with_0 = "key_index_with_0.txt"

indexex_0 = []
indexex_1 = []
if not os.path.exists(key_index_file_path):
    print(f'{key_index_file_path} 路径不存在！') 
else:
    file_0 = key_index_file_path + key_index_with_0
    file_1 = key_index_file_path + key_index_with_1
    
    with open(file_0, mode='r',encoding='utf-8') as f:
        for line in tqdm(f):
            index = int(line.strip())
            indexex_0.append(index)

    with open(file_1, mode='r',encoding='utf-8') as f:
        for line in tqdm(f):
            index = int(line.strip())
            indexex_1.append(index)
    
    indexex_0 = list(set(indexex_0))        
    indexex_1 = list(set(indexex_1))

data.loc[indexex_0,"label"] = 0
data.loc[indexex_1,"label"] = 1

data[["id", "label"]].to_csv(final_result_data_path, index=None, sep=",")

print(data['label'].value_counts())
a = 1