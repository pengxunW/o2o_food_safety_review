import argparse

# import os.path
import time

def parse_args():
    same_time = time.strftime("%m-%d_%H-%M")

    parser = argparse.ArgumentParser(description="Final Code for 2022BDCI RE")

    parser.add_argument("--seed", type=int, default=2024, help="random seed.")
    # 选择 cup 还是 gpu
    parser.add_argument("--device", type=str, default="cuda")

    ## 交叉验证的折数
    parser.add_argument("--k_fold", type=int, default=5)
    """这里将来删掉"""
    # ## 用于小样本与大样本之间的区分标志
    # parser.add_argument("--data_count", type=str, default="samll", help="data count")

    # (1) ========================= Data Configs ==========================

    parser.add_argument("--dropout", type=float, default=0.3, help="dropout ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1)

    # (2) ========================= Model Configs ==========================

    # token序列的最大长度
    parser.add_argument("--max_seq_len", type=int, default=100)

    """这里可以改成 folder ＋ 对应的文件保存路径"""
    # 加载预训练模型的路径
    parser.add_argument(
        "--bert_pred",
        type=str,
        default="./pretrained_models/uerroberta-base-finetuned-dianping-chinese",
        help="bert ptetrained model",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20, help="How many epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--learning_rate", default=15e-6, type=float, help="initial learning rate"
    )
    # 若超过该 batch 数效果还没提升，则提前结束训练
    parser.add_argument("--require_improvement", type=int, default=1)

    # 以上需调整------------------------------------------------------------------
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight deay if we apply some."
    )
    """这个参数是啥意思， 用在 optimizer()"""
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--early_stop", default=5, type=int)

    """这里得修改"""
    # (3) ========================== Small Sample Config =============================

    # 用于方便调试函数的小样本训练集、验证集和测试集
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/sample_train_data.csv",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="./data/sample_valid_data.csv",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/sample_test_data.csv",
    )
    # 小样本超参数保存文件夹
    parser.add_argument(
        "--args_folder",
        type=str,
        default=f"./args/{same_time}/",
    )
    # 小样本日志保存文件夹
    parser.add_argument(
        "--log_folder",
        type=str,
        default=f"./logs/{same_time}/",
    )
    # 小样本 tensorboard 可视化文件夹
    parser.add_argument(
        "--tensorboard_folder",
        type=str,
        default=f"./tensorboard/{same_time}/",
    )
    # 小样本模型保存文件夹
    # 模型不添加时间信息用于区别，直接选择覆盖，因为 args 已经保存
    parser.add_argument(
        "--model_save_folder",
        type=str,
        default="./models/",
    )
    # 小样本结果保存文件夹
    parser.add_argument(
        "--origin_result_folder",
        type=str,
        default=f"./result/origin_result/{same_time}/",
    )
    parser.add_argument(
        "--final_result_folder",
        type=str,
        default=f"./result/final_result/{same_time}/",
    )


    # 以上需调整------------------------------------------------------------------
    
    # parser.add_argument('--input_size', type=int, default=128)           # bert 中暂时没用到

    # parser.add_argument("--out_channels", type=int, default=250)  # TextCnn 的卷积输出
    # parser.add_argument(
    #     "--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小"
    # )
    # parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
    # parser.add_argument('--num_classes', type=int, default=1)
    # parser.add_argument('--stride', type=int, default=2)

    # parser.add_argument("--kernel_size", type=int, default=3)

    # parser.add_argument(
    #     "--select_model_last",
    #     type=bool,
    #     default=True,
    #     help="选择模型 BertTextModel_last_layer",
    # )
    # parser.add_argument(
    #     "--save_model_best",
    #     type=str,
    #     default=os.path.join("models/chinese-roberta-wwm-ext-large", "best_model.pt"),
    # )  # 最佳的    模型
    # parser.add_argument(
    #     "--save_model_last",
    #     type=str,
    #     default=os.path.join("models/chinese-roberta-wwm-ext-large", "last_model.pt"),
    # )  # 最后一次保存的模型
    return parser.parse_args()


# def parse_args():
#     same_time = time.strftime("%m-%d_%H-%M")

#     parser = argparse.ArgumentParser(description="Final Code for 2022BDCI RE")

#     parser.add_argument("--seed", type=int, default=2024, help="random seed.")
#     # 选择 cup 还是 gpu
#     parser.add_argument("--device", type=str, default="cuda")

#     ## 交叉验证的折数
#     parser.add_argument("--k_fold", type=int, default=5)
#     '''将来可以删掉'''
#     ## 用于小样本与大样本之间的区分标志
#     # parser.add_argument("--data_count", type=str, default="samll", help="data count")

#     # (1) ========================= Data Configs ==========================

#     parser.add_argument("--dropout", type=float, default=0.3, help="dropout ratio")
#     parser.add_argument("--val_ratio", type=float, default=0.1)

#     # (2) ========================= Model Configs ==========================

#     # token序列的最大长度
#     parser.add_argument("--max_seq_len", type=int, default=100)

#     """这里可以改成 folder ＋ 对应的文件保存路径"""
#     # 加载预训练模型的路径
#     '''这里得更改'''
#     parser.add_argument(
#         "--bert_pred",
#         type=str,
#         default="./pretrained_models/uerroberta-base-finetuned-dianping-chinese",
#         help="bert ptetrained model",
#     )
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--num_epochs", type=int, default=20, help="How many epochs")
#     parser.add_argument("--warmup_ratio", type=float, default=0.1)
#     parser.add_argument(
#         "--learning_rate", default=15e-6, type=float, help="initial learning rate"
#     )
#     # 若超过该 batch 数效果还没提升，则提前结束训练
#     parser.add_argument("--require_improvement", type=int, default=1)

#     # 以上需调整------------------------------------------------------------------
#     parser.add_argument("--hidden_size", type=int, default=768)
#     parser.add_argument("--num_classes", type=int, default=2)

#     parser.add_argument(
#         "--weight_decay", default=0.01, type=float, help="Weight deay if we apply some."
#     )
#     """这个参数是啥意思， 用于 optimizer()"""
#     parser.add_argument(
#         "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
#     )
#     parser.add_argument("--early_stop", default=5, type=int)

#     """这里得修改"""

#     # (3) ========================== All Sample Config =============================
#     # 用于生成最终模型结果的大样本训练集和测试集
#     parser.add_argument(
#         "--train_data_path",
#         type=str,
#         default="./data/train.csv",
#     )
#     parser.add_argument(
#         "--test_data_path",
#         type=str,
#         default="./data/test_new.csv",
#     )
#     # 大样本超参数保存文件夹
#     parser.add_argument(
#         "--args_folder",
#         type=str,
#         default=f"./all_sample/args/{same_time}/",
#     )
#     # 大样本日志保存文件夹
#     parser.add_argument(
#         "--log_folder",
#         type=str,
#         default=f"./all_sample/logs/{same_time}/",
#     )
#     # 大样本 tensorboard 可视化文件夹
#     parser.add_argument(
#         "--tensorboard_folder",
#         type=str,
#         default=f"./all_sample/tensorboard/{same_time}/",
#     )
#     # 大样本模型保存文件夹
#     # 模型不添加时间信息用于区别，直接选择覆盖，因为 args 已经保存
#     parser.add_argument(
#         "--model_save_folder",
#         type=str,
#         default="./all_sample/models/",
#     )
#     # 大样本结果保存文件夹
#     parser.add_argument(
#         "--origin_result_folder",
#         type=str,
#         default=f"./all_sample/result/origin_result/{same_time}/",
#     )
#     parser.add_argument(
#         "--final_result_folder",
#         type=str,
#         default=f"./all_sample/result/final_result/{same_time}/",
#     )

#     # 以上需调整------------------------------------------------------------------

#     # parser.add_argument('--input_size', type=int, default=128)           # bert 中暂时没用到

#     # parser.add_argument("--out_channels", type=int, default=250)  # TextCnn 的卷积输出
#     # parser.add_argument(
#     #     "--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小"
#     # )
#     # parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
#     # parser.add_argument('--num_classes', type=int, default=1)
#     # parser.add_argument('--stride', type=int, default=2)

#     # parser.add_argument("--kernel_size", type=int, default=3)

#     # parser.add_argument(
#     #     "--select_model_last",
#     #     type=bool,
#     #     default=True,
#     #     help="选择模型 BertTextModel_last_layer",
#     # )
#     # parser.add_argument(
#     #     "--save_model_best",
#     #     type=str,
#     #     default=os.path.join("models/chinese-roberta-wwm-ext-large", "best_model.pt"),
#     # )  # 最佳的    模型
#     # parser.add_argument(
#     #     "--save_model_last",
#     #     type=str,
#     #     default=os.path.join("models/chinese-roberta-wwm-ext-large", "last_model.pt"),
#     # )  # 最后一次保存的模型

#     return parser.parse_args()
