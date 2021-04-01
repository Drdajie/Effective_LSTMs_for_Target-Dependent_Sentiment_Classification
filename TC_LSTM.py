# enconding:utf-8
# file：LSTM.py
# author：Drdajie
# email：drdajie@gmail.com

import random
import torch
import torch.nn as nn
from Tool_and_Layor import Dynamic_LSTM

class TC_LSTM(nn.Module):
    def __init__(self,embedding_matrix,opt):
        super(TC_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = Dynamic_LSTM(opt.input_dim*2,opt.hidden_dim,
                            num_layers=opt.num_layer, batch_first=opt.batch_first)
        self.lstm_r = Dynamic_LSTM(opt.input_dim*2,opt.hidden_dim,
                            num_layers=opt.num_layer, batch_first=opt.batch_first)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.num_class)

    def forward(self, inputs):
        # 预处理
        x_l, x_r,target = inputs['x_left'], inputs['x_right'],inputs['aspect']
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1, dtype=torch.float)[:, None, None]
                        #因为 target 中可能有多个词需要求平均，所以需要元素类型为 float
        x_l, x_r, target = self.embed(x_l), self.embed(x_r), self.embed(target)
        # input seq 拼接 aspect 信息
        v_target = torch.div(target.sum(dim=1, keepdim=True),
                             target_len)  # v_{target} in paper: average the target words
        x_l = torch.cat((x_l, torch.cat(([v_target] * x_l.shape[1]), 1)),2)
        x_r = torch.cat((x_r, torch.cat(([v_target] * x_r.shape[1]), 1)),2)
        # LSTM 层
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        # 合并 & Linear 层
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out