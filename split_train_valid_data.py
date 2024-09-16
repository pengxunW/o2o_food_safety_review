import pandas as pd
from sklearn.model_selection import train_test_split
from config import parse_args

'''make_sample_data 用于构造小规模的训练集、验证集和测试集，方便调试函数'''
def make_sample_data(args):
    full_train_path =   args.full_train_data_path
    test_data_path = args.test_data_path
    train_data = pd.read_csv(full_train_path, sep='\t')
    test_data = pd.read_csv(test_data_path)
    
    # train_data, valid_data, test_data 各抽取 100 条用于方便调试函数
    # 即使 train_data 与 valid_data 部分重复也可以，因为我们主要是测试项目的整个流程
    sample_train_data = train_data.sample(frac=0.01, random_state=2024, axis=0)
    sample_valid_data = train_data.sample(frac=0.01, random_state=2025, axis=0)
    sample_test_data = test_data.sample(frac=0.05, random_state=2024, axis=0)
    
    sample_train_data.to_csv('data/sample_train_data.csv',sep=',', index=False)
    sample_valid_data.to_csv('data/sample_valid_data.csv', sep=',',index=None)
    sample_test_data.to_csv('data/sample_test_data.csv',sep=',', index=False)
    return

if __name__=='__main__':
        
    args = parse_args()
    make_sample_data(args=args)
    # full_train_path =   args.full_train_data_path
    # train_data_path = args.train_data_path
    # dev_data_path = args.dev_data_path

    # data = pd.read_csv(full_train_path, sep='\t')

    # train_data, dev_data = train_test_split(data, test_size = 0.18, random_state=42 )

    # columns = ['label','comment']

    # # columns = data.columns
    # # train_data.loc[0, 'comment']
    # # dev_data.loc[0, 'comment']
    # train_data.to_csv(train_data_path, sep=',', index=False)
    # dev_data.to_csv(dev_data_path, sep=',', index=False)

