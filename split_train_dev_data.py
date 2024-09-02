import pandas as pd
from sklearn.model_selection import train_test_split
from config import parse_args

args = parse_args()


full_train_path =   args.full_train_data_path
train_data_path = args.train_data_path
dev_data_path = args.dev_data_path

data = pd.read_csv(full_train_path, sep='\t')

train_data, dev_data = train_test_split(data, test_size = 0.18, random_state=42 )

columns = ['label','comment']

# columns = data.columns
# train_data.loc[0, 'comment']
# dev_data.loc[0, 'comment']
train_data.to_csv(train_data_path, sep=',', index=False)
dev_data.to_csv(dev_data_path, sep=',', index=False)

