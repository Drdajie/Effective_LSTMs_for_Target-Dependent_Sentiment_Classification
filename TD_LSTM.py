# enconding:utf-8
# file：LSTM.py
# author：Drdajie
# email：drdajie@gmail.com

import torch
import torch.nn as nn
from Tool_and_Layor import Dynamic_LSTM

class TD_LSTM(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(TD_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = Dynamic_LSTM(opt.input_dim,opt.hidden_dim,
                                   num_layers=opt.num_layer, batch_first=opt.batch_first)
        self.lstm_r = Dynamic_LSTM(opt.input_dim, opt.hidden_dim,
                                   num_layers=opt.num_layer, batch_first=opt.batch_first)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.num_class)

    def forward(self, inputs):
        x_l, x_r = inputs['x_left'], inputs['x_right']
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out