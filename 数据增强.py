import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import parse_args

args = parse_args()

result_data_with_comment_path = args.result_save_path_with_comment

data = pd.read_csv(result_data_with_comment_path, sep=',')

key_word=['蚊子','剩饭','剩菜','剩下的','剩的','不新鲜','没熟','老鼠','烂','骚味',\
        '拉肚子','苍蝇','虫','臭','想吐的','太硬','头发','恶心']

# key_word = ['不卫生','糟糕','一股','虫子','坏','吐','#','很','…','死的']

# key_word = ['不卫生']

# key_word = ['糟糕']

# key_word = ['一股']
index = []
for i in key_word:
    index_i = data[data['comment'].str.contains(i)].index
    index.extend(index_i)
    # print (data.loc[index,['comment',"label"]])

columns = ['comment','label']
key_data = data.loc[index,columns]
key_data = key_data.drop_duplicates()
key_data = key_data[key_data.label == 0]
key_data.to_csv('data/数据增强/key_data_1.csv',sep = ",", index=True, index_label='index')


except_index = [1753, 429, 1350, 418, 1090]
index = key_data.index.to_list()

key_index_file_path = 'data/数据增强/'
key_index_file = 'key_index.txt'
 
if not os.path.exists(key_index_file_path):
    os.makedirs(key_index_file_path)
    print(f"{key_index_file_path} 文件路径创建成功！")
    with open(key_index_file, mode='w',encoding='utf-8') as f:
        for i in index:
            if i not in except_index:
                f.write(str(i) + '\n') 
else:
    with open(key_index_file, mode='w',encoding='utf-8') as f:
        for i in index:
            if i not in except_index:
                f.write(str(i) + '\n') 
        
        
        
        
# data.loc[index,"label"] = 1

# data[["id", "label"]].to_csv(args.result_save_path, index=None, sep=",")

  
# print(data['label'].value_counts())
a = 1