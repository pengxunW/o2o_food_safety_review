# ccf O2O商铺食品安全相关评论发现

## 1.1 怎么运行

1. 使用 conda 新创建一个 python==3.8.10 的虚拟环境，之后安装相应的包， pip install -r requirements.txt
2. 运行 main.py，在 models/chinese-roberta-wwm-ext-large/路径下得到True_best_model.pt
3. 运行 predict_with_comment.py 在 data/result 得到  chinese-roberta-wwm-ext-large_result_with_comment.csv
4. 运行 final_result.py 在data/result 得到 chinese-roberta-wwm-ext-large_result.csv 提交线上 F1_score： 0.9282
5. 模型调参在 config.py
6. 除了在终端中查看运行时的情况之外，还可以在开启一个终端，键入 tensorboard --logdir=log/chinese-roberta-wwm-ext-large, 点击进入网址，可在线可视化查看模型的运行情况。
7. 最优结果即是当前文件，可视化结果是在log/chinese-roberta-wwm-ext-large/09-03_13.45/events.out.tfevents.1725342314.autodl-container-f11a41911a-67b4ca11.13877.0

## 其他

1. 可以逐步打上断点进行调试，观察关键变量每一步的变化，看懂比较耗时。
2. 本部分代码默认运行的中文预训练模型是chinese-roberta-wwm-ext-large，可以在 huggingface 上下载。
