import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


#########构建DataSet########
class MyDataSet(Dataset):
    def __init__(self, data, args, mode="train"):
        self.df = data
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_pred)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 对文本进行编码，最大长度为128
        max_length = self.args.max_seq_len
        device = self.args.device
        text = self.df.loc[index, "comment"]
        """这里 max_len 将来是否可以添加到 args 里面？同时可以设置 device =  self.args.device，减少代码冗余度"""
        text_encode = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # 取出input_ids,attention_mask
        input_ids = text_encode["input_ids"].to(device).squeeze()
        attention_mask = text_encode["attention_mask"].to(device).squeeze()

        if self.mode == "train":
            label = torch.tensor(self.df.loc[index, "label"], device=device)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,
            }
        elif self.mode == "test":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
