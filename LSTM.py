# enconding:utf-8
# file：LSTM.py
# author：Drdajie
# email：drdajie@gmail.com

import torch
import torch.nn as nn
from Tool_and_Layor import Dynamic_LSTM
import Tool_and_Layor
from Hyper_Parameters import *


class LSTM(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(LSTM, self).__init__()
        self.embedding_matrix = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_layor = \
            Dynamic_LSTM(opt.input_dim,opt.hidden_dim,opt.num_layer,bias=opt.bias,batch_first=True)
        self.linear_layor = nn.Linear(in_features=opt.hidden_dim,bias=opt.bias,out_features=opt.num_class)

    def forward(self,samples):
        """
        samples -> get embedding and length
               ->  sort -> pad and pack ->process using RNN -> unpack ->unsort
               -> linear layor
        :param inputs: some sequence's indices
        :return: 一个矩阵，其 shape 为 batch × n_class
        """
        inputs = samples['x']
        x = self.embedding_matrix(inputs)        #x.shape = batch_size,sequence_len,embedding_dim
        x_len = torch.sum(inputs != 0,dim=-1)    #因为 inputs中含有 paded mark
                                                 # -> 所以用 x_len记录 x 中每句话中有效的 word 个数
        x = Tool_and_Layor.squeeze_sequence(x,x_len)
        #1_进行 lstm layor 处理
        ot,(ht,_) = self.lstm_layor(x,x_len)
        #2_进行 linear layor 处理
        output = self.linear_layor(ht[0])
        return output

