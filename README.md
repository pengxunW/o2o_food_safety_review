# ccf O2O商铺食品安全相关评论发现

## 1. 项目结构

整个项目的核心文件夹有两个，一个是 **small_sample/**，另一个是 **all_sample/**。

两个文件夹下的代码均可单独运行，唯一不同的点是 small_sample/ 文件夹下的数据是从原始的全量数据中抽取部分，从而方便在本地调试，跑通代码的逻辑结构，(由于本地的算力支持有限)。除此之外，两个文件夹下的代码逻辑结构均相同。

！！！**注意：由于两个文件夹均可以单独看作是独立的完整的项目，在 vscode 中运行时要选择单独的一个文件作为工作目录，否则，会出现相对路径的错误。**

以在本地运行小样本数据为例，具体方式是：在 vscode 中，依次点击 File --> Open Folder --> small_sample，就可以选择 small_sample 作为当前的工作目录。

## 2. 怎么运行

待完善......(9-21)

1. 使用 conda 新创建一个 python==3.8.10 的虚拟环境，之后安装相应的包， pip install -r requirements.txt
<!-- 2. 运行 main.py，在 models/chinese-roberta-wwm-ext-large/路径下得到True_best_model.pt
3. 运行 predict_with_comment.py 在 data/result 得到  chinese-roberta-wwm-ext-large_result_with_comment.csv
4. 运行 final_result.py 在data/result 得到 chinese-roberta-wwm-ext-large_result.csv 提交线上 F1_score： 0.9282
5. 模型调参在 config.py
6. 除了在终端中查看运行时的情况之外，还可以在开启一个终端，键入 tensorboard --logdir=log/chinese-roberta-wwm-ext-large, 点击进入网址，可在线可视化查看模型的运行情况。
7. 最优结果即是当前文件，可视化结果是在log/chinese-roberta-wwm-ext-large/09-03_13.45/events.out.tfevents.1725342314.autodl-container-f11a41911a-67b4ca11.13877.0 -->

<!-- ## 2. 注意事项

1. 可以逐步打上断点进行调试，观察关键变量每一步的变化。 -->

## 3. 开发日志

### 3.1. 重构代码(2024-09-02)

#### 3.1.1 能不能构建一个比较小的数据集用于快速理清代码逻辑结构 (9-2)

~~开发中...~~  9-16 完成：在 split_train_valid_data.py 中。

#### 3.1.2. 将模型的训练过程模块化 (9-2)

~~开发中...~~ 9-16 日已完成。

具体来说，在模型训练时，并没有记录相关的指标 (metrics)，比如损失(loss)、准确率(accuracy)、F1_score。在模型训练完成一轮之后(per epoch)，专门调用验证函数 (valid_one_epoch) 来输出一轮的 metrics，并根据这些指标来选择是否更新最佳模型，之后将这些 metrics 使用日志 (log) 记录成文件，同时在控制台上 (console) 上打印。

### 3.2. 添加额外的功能(2024-09-03)

#### 3.2.1. 输出模型训练过程中的log信息 (9-3)

9-16 完成。

将 python 的内置日志记录模块 logging 封装成一个 Logger 类，在主函数中实例化之后保存和输出日志信息。

#### 3.2.2. 采用 bert-large模型训练一轮 (9-3)

开发中...

#### 3.2.3. 添加项目的完整运行说明 (9-16)

待完善...
