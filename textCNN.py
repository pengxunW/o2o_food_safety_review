# from transformers import BertModel
# import torch.nn.functional as F

# def conv_and_pool(self, x, conv):
#     x = F.relu(conv(x)).squeeze(3)  #[batch_size, out_channels, output_length]
#     x = F.max_pool1d(x, x.size(2)).squeeze(2)   #[batch_size, channels]
#     return x

# num_filters = 256
# filter_sizes = (2, 3, 4)
# convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.hidden_size))
#              for k in config.filter_sizes])
             
# bert=BertModel.from_pretrained('bert-base-chinese')

# encoder_out = self.bert(context, attention_mask=mask).last_hidden_state   #[batch_size, sequence_length, hidden_size]
# out = encoder_out.unsqueeze(1) # [batch_size, 1(in_channels), sequence_length, hidden_size]
# out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  #[batch_size, channels*len(self.convs)]


