import argparse
import os.path

def parse_args():
    parser = argparse.ArgumentParser(description="Final Code for 2022BDCI RE")

    parser.add_argument("--seed", type=int, default=2024, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--full_train_data_path', type=str, default='./data/train.csv')
    parser.add_argument('--train_data_path', type=str, default='./data/train_data.csv')
    parser.add_argument('--dev_data_path', type=str, default='./data/dev_data.csv')
    parser.add_argument('--test_data_path', type=str, default='./data/test_new.csv')
    # parser.add_argument('--result_save_path', type=str, default='./data/result/base_bert_result.csv')
    # parser.add_argument('--result_save_path_with_comment', type=str, default='./data/result/bert_TextCNN_result_with_comment.csv')
    # parser.add_argument('--result_save_path', type=str, default='./data/result/bert_TextCNN_result.csv')
    parser.add_argument('--result_save_path_with_comment', type=str, default='./data/result/chinese-roberta-wwm-ext-large_result_with_comment.csv')
    parser.add_argument('--result_save_path', type=str, default='./data/result/chinese-roberta-wwm-ext-large_result.csv')
    parser.add_argument('--val_ratio', type=float, default=0.1)

    # ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_epochs', type=int, default=50, help='How many epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--learning_rate', default=15e-6, type=float, help='initial learning rate')
    
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    '''这个参数是啥意思， 用在 optimizer()'''
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--early_stop", default=5, type=int)



    '''这里得修改'''
    # ========================== Model Config =============================
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=100)
  
    parser.add_argument("--bert_pred", type=str, default="hfl/chinese-roberta-wwm-ext-large", help="bert 预训练模型")
    # parser.add_argument("--bert_pred", type=str, default="hfl/chinese-bert-wwm", help="bert 预训练模型")
    # parser.add_argument("--bert_pred", type=str, default="bert-base-chinese", help="bert 预训练模型")
    
    # parser.add_argument('--input_size', type=int, default=128)           # bert 中暂时没用到
    
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--out_channels', type=int, default= 250)            #TextCnn 的卷积输出
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    # parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
    # parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--require_improvement', type=int, default=400) # 若超过400 batch效果还没提升，则提前结束训练
    
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--model_save_path', type=str, default='./models/base_bert/base_bert_model.pt')
    parser.add_argument('--log_path', type=str, default='./log/chinese-roberta-wwm-ext-large')
    # parser.add_argument('--log_path', type=str, default='./log/bert_TextCNN')
    
    parser.add_argument("--select_model_last", type=bool, default=True, help="选择模型 BertTextModel_last_layer")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("models/chinese-roberta-wwm-ext-large", "best_model.pt"))     # 最佳的    模型
    parser.add_argument("--save_model_last", type=str, default=os.path.join("models/chinese-roberta-wwm-ext-large", "last_model.pt"))     #最后一次保存的模型

    return parser.parse_args()
