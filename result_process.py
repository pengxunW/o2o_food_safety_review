import os
import pandas as pd
from tqdm import tqdm
from config import parse_args

args = parse_args()

result_data_path = args.result_save_path

result_data_with_comment_path = args.result_save_path_with_comment
data = pd.read_csv(result_data_with_comment_path, sep=',')

# key_index_file_path = 'data/数据增强/'
# key_index_file = 'key_index.txt'

# indexex = []
# if not os.path.exists(key_index_file_path):
#     print(f'{key_index_file_path} 路径不存在！') 
# else:
#     file = key_index_file_path + key_index_file
#     with open(file, mode='r',encoding='utf-8') as f:
#         for line in tqdm(f):
#             index = int(line.strip())
#             indexex.append(index)

# indexex = list(set(indexex))
# data.loc[indexex,"label"] = 1
data[["id", "label"]].to_csv(args.result_save_path, index=None, sep=",")

print(data['label'].value_counts())
a = 1