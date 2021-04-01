# enconding:utf-8
# file：LSTM.py
# author：Drdajie
# email：drdajie@gmail.com

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import Data_Utility as data_util
from sklearn import metrics
from Tool_and_Layor import Dynamic_LSTM
import math

#将类当作命名空间用
class file_args:
    train_path = "./Data/train.raw"  # the path of traning data
    test_path = "./Data/test.raw"  # the path of test data
class preTrain_args:        #主要的超参数
    train_ratio = 0.8       #the size ratio of train
    validation_ratio = 0.2  #the size ratio of validation
    embedding_dim = 100     #dimension of word embedding
    max_seq_len = 85
    paded_mark = 0
    seed = int(1234)
    uniform_range = 0.003
    l2reg = 0.001
class model_args:
    model_name = 'TC-LSTM'
    input_dim = preTrain_args.embedding_dim
    hidden_dim = 300
    num_layer = 1
    bias = True
    batch_first = True
    num_class = 3
class train_args:
    epoch = 30
    learn_rate = 0.01       #learning rate
    batch_size = 64         #the mini-batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 5
    """clipping_threshold = 200
    norm_type = 2           #衡量距离时用的范数类型"""

class TC_LSTM(nn.Module):
    def __init__(self,embedding_matrix):
        super(TC_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = Dynamic_LSTM(model_args.input_dim*2,model_args.hidden_dim,
                            num_layer=model_args.num_layer, batch_first=model_args.batch_first)
        self.lstm_r = Dynamic_LSTM(model_args.input_dim*2,model_args.hidden_dim,
                            num_layer=model_args.num_layer, batch_first=model_args.batch_first)
        self.dense = nn.Linear(model_args.hidden_dim*2, model_args.num_class)

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
        _, h_n_l, _ = self.lstm_l(x_l, x_l_len)
        _, h_n_r, _ = self.lstm_r(x_r, x_r_len)
        # 合并 & Linear 层
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out

class Performer:
    def __init__(self):
        """
        为训练做准备
        """
        # data
        self.prepare_data()
        # random seed
        self.set_seed()
        # model
        # 1) instance model
        self.my_model = TC_LSTM(embedding_matrix=self.embedding_matrix).to(train_args.device)
        # 2) initial model parameters
        self.initial_parameters()
        # loss and optimizer
        #self.ce_lossFunc = nn.CrossEntropyLoss().to(train_args.device)   # 见收藏夹
        self.nll_loss = nn.NLLLoss().to(train_args.device)
        params = filter(lambda p: p.requires_grad, self.my_model.parameters())
        self.optimizer = optim.Adam(lr=train_args.learn_rate,params=params,weight_decay=preTrain_args.l2reg)

    def initial_parameters(self):
        for child in self.my_model.children():
            for p in child.parameters():
                if p.requires_grad:
                    torch.nn.init.uniform_(p,-preTrain_args.uniform_range,preTrain_args.uniform_range)

    def prepare_data(self):
        """
        初始化数据：
            1_生成一个词汇表（包含所有本次实验要用到的word），为每个word找到一个对应的 index
            2_生成 embedding matrix，matrix 每一行代表一个 word 的 vector。对应关系为 matrix
            中word 对应的行的 index 与其在 vocabulary 中的 index 相同。
            3_得到 training、validation、testing 三者的 Dataloader
        :return:
        """
        # 1_构建词汇表 和 embedding matrix
        # 1.1_构建词汇表
        self.vocabulary = data_util.Vocabulary()
        self.vocabulary.take_into_vocabulary_form_text(file_args.train_path)  # 将 train data 中的 tokens 加入词汇表中
        self.vocabulary.take_into_vocabulary_form_text(file_args.test_path)  # 将 test data 中的 tokens 加入词汇表中
        # 1.2_构建 embedding matrix
        self.embedding_matrix = data_util.Vectorizer(self.vocabulary,preTrain_args.embedding_dim).\
            generate_embeddingMatrix()
        self.embedding_matrix = self.embedding_matrix.to(train_args.device)
        # 2_得到 Dataloader
        # 2.1_构建 Dataset
        # 1) 得到所有数据
        train_val_dataset = data_util.LSTM_DataSet(file_args.train_path, self.vocabulary,mode=model_args.model_name)
        self.test_dataset = data_util.LSTM_DataSet(file_args.test_path, self.vocabulary,mode=model_args.model_name)
        # 2) 划分数据集
        train_set_size = int(len(train_val_dataset) * preTrain_args.train_ratio)
        val_set_size = len(train_val_dataset) - train_set_size
        self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_set_size, val_set_size])
        # 2.2_构建 Dataloader
        self.train_dataLoader = DataLoader(self.train_dataset, train_args.batch_size, shuffle=True)
        self.val_dataLoader = DataLoader(self.val_dataset, train_args.batch_size, shuffle=False)
        self.test_dataLoader = DataLoader(self.test_dataset, train_args.batch_size, shuffle=False)

    def set_seed(self,seed=preTrain_args.seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def train(self):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        #整个训练
        for i_epoch in range(train_args.epoch):
            print('epoch：', i_epoch)
            n_correct, n_total, loss_total = 0., 0., 0.
            self.my_model.train()             # switch model to training mode
            #更新参数阶段（tarin_dataset）
            for i_batch,batch in enumerate(self.train_dataLoader):
                self.optimizer.zero_grad()
                outputs = self.my_model(batch)#.reshape(1,-1)
                #targets = batch['y'].to(train_args.device)
                outputs = F.softmax(outputs)
                targets = batch['y'].to(train_args.device)
                #outputs = 1/outputs
                #for i in range(len(targets)):
                #    torch.clamp_(outputs[i,targets[i]],min=1/200)
                torch.clamp_(outputs,min = 1/200, max = 1)
                #outputs = 1/outputs
                outputs = torch.log(outputs)
                loss = self.nll_loss(outputs,targets)
                #loss = self.ce_lossFunc(outputs,targets)
                loss.backward()
                # nn.utils.clip_grad_norm_(self.my_model.parameters(),
                                         # max_norm=train_args.clipping_threshold,
                                         # norm_type=train_args.norm_type)
                self.optimizer.step()
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if i_batch % 10 == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print('train_loss',train_loss,'   train_acc',train_acc)
            #选择阶段（validation）
            val_acc, val_f1 = self._evaluate_acc_f1(self.val_dataLoader)
            print('val_acc：', val_acc)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= train_args.patience:
                print('>> early stop.')
                break
        print('最高准确率：',max_val_acc)
        print('最高 f1 score 为：',max_val_f1)

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        targets_all, outputs_all = None, None
        # switch model to evaluation mode
        self.my_model.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                targets = batch['y'].to(train_args.device)
                outputs = self.my_model(batch)

                n_correct += (torch.argmax(outputs,dim=-1) == targets).sum().item()
                n_total += len(outputs)

                if targets_all is None:
                    targets_all = targets
                    outputs_all = outputs
                else:
                    targets_all = torch.cat((targets_all, targets), dim=0)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def test(self):
        test_acc, test_f1 = self._evaluate_acc_f1(self.test_dataLoader)
        print('测试集准确率为：',test_acc)
        print('测试集 f1 score 为：',test_f1)

    def run(self):
        self.train()
        self.test()

def main():
    performer = Performer()
    performer.run()

if __name__ == "__main__":
    main()