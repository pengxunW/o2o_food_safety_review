import pandas as pd
from config import parse_args

args = parse_args()

result_with_commnet_data_path = args.result_save_path_with_comment
result_data_path = args.result_save_path

'''原始预测结果的标签统计'''
data1 = pd.read_csv(result_with_commnet_data_path, sep=',')
data1['label'].value_counts()
print(data1['label'].value_counts())

'''数据增强后预测结果的标签统计'''
data2 = pd.read_csv(result_data_path, sep=',')
data2['label'].value_counts()
print(data2['label'].value_counts())
a = 1