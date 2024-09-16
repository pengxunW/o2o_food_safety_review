
import torch 
from torch import nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import AutoConfig,AutoModel

# class Model_LSTM(nn.Module):
    
#     '''这里模型初始化时要传入之前实例化后的参数变量'''
#     def __init__(self,args):
#         super().__init__()
        
#         self.args = args
#         #定义LSTM
#         '''这里 num_layers 定义为 1 或 2都可，不过要重新设计对应的损失函数'''
#         self.encoder = nn.LSTM(input_size=self.args.input_size,hidden_size=self.args.hidden_size,num_layers=1,bidirectional =True,batch_first=True)
#         #定义分类的线性层
#         self.dence = nn.Linear(self.args.hidden_size*2,2)
#         #定义激活函数relu
#         self.active = nn.ReLU()
#         #定义交叉熵损失
#         self.loss_fnc = torch.nn.CrossEntropyLoss()
        
#     def forward(self,inputs,mode ='train'):
        
#         embedding = inputs['embedding']
#         #将embedding输入lstm，得到lstm的输出
#         encode_output, (hn,cn) = self.encoder(embedding)
        

#         #将隐藏层正向反向拼接        
#         output = torch.cat([hn[-1],hn[-2]],-1)
#         #将拼接结果经过激活函数后送入分类的线性层,得到模型输出
#         output = self.dence(self.active(output))
        
#         #在推理模式下，输出模型输出
#         if mode!='train':
#             return output
        
#         #训练模式下，输出损失
#         loss = self.loss_fnc(output,inputs['label'].view(-1))
#         return loss
    
    
'''这里模型没有添加 dropout()'''

#定义模型
class baseModel(nn.Module):
    def __init__(self, args):
        super(baseModel, self).__init__()
        self.bert_pred = args.bert_pred
        '''模型名称将来也可以在 config 中定义'''
        config = AutoConfig.from_pretrained(self.bert_pred)
        self.bert = AutoModel.from_pretrained(self.bert_pred,config=config)

        self.out=nn.Linear(config.hidden_size,args.num_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_out=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #获取bert的Pool向量
        pooler=bert_out['pooler_output']
        #下游分类层
        out=self.out(pooler)
        #转换为概率
        
        '''这里可能需要更加详细一点，损失函数呢，还有模型的训练和评估部分的逻辑'''
        '''如果采用sigmoid()'''
        # out=torch.sigmoid(out)
        # out = out.squeeze()
        '''采用 softmax(), 使用 nn.crossentropyloss() 里面有softmax(),
        不需要单独再将模型的预测结果再次 softmax() 一下'''
        # out = F.softmax(out, dim=1)
        return out
    
'''这个模型是 bert + TextCNN, 将 bert 最后一层 hidden_state 作为 TextCNN 的输入'''  
class BertTextModel_last_layer(nn.Module):
    def __init__(self,args):
        super(BertTextModel_last_layer, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(self.args.bert_pred)
        for param in self.bert.parameters():
            param.requires_grad = True

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.args.out_channels, kernel_size=(k, self.args.hidden_size),) for k in self.args.filter_sizes]
        )
        self.dropout = nn.Dropout(self.args.dropout)
        self.fc = nn.Linear(args.out_channels * len(self.args.filter_sizes), self.args.num_classes) # 一种类型的一个卷积核filter 对应一个 outchannel 特征

    def conv_pool(self, x, conv):
        x = conv(x)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1, 1]
        x = F.relu(x)
        x = x.squeeze(3)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1]
        size = x.size(2)
        x = F.max_pool1d(x, size)   # shape[batch_size, out_channels, 1]
        x = x.squeeze(2)  # shape[batch_size, out_channels]
        return x

    def forward(self, input_ids, attention_mask):           # shape [batch_size, max_len] 
        
        hidden_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        out = hidden_out.last_hidden_state.unsqueeze(1)   # shape [batch_size, 1, max_len, hidden_size]
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1)  # shape  [batch_size, self.args.out_channels * len(self.args.filter_sizes]
        out = self.dropout(out)
        out = self.fc(out)
        
        out=torch.sigmoid(out)
        out = out.squeeze()
        return out


        
