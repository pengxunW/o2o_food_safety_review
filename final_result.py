import os
import pandas as pd
from tqdm import tqdm
from config import parse_args

args = parse_args()

result_data_path = args.result_save_path
result_data_with_comment_path = args.result_save_path_with_comment

data = pd.read_csv(result_data_with_comment_path, sep=',')

data[["id", "label"]].to_csv(args.result_save_path, index=None, sep=",")

print(data['label'].value_counts())
a = 1