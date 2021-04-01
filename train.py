from Hyper_Parameters import *
from Data_Utility import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import random
import numpy
from time import strftime,localtime
from sklearn import metrics
import torch
from LSTM import LSTM
from TD_LSTM import TD_LSTM
from TC_LSTM import TC_LSTM
import math

class Performer:
    def __init__(self):
        """
        为训练做准备
        """
        # data
        self.prepare_data()
        # model
        # 1) instance model
        model = {'LSTM': LSTM, 'TD-LSTM': TD_LSTM, 'TC-LSTM': TC_LSTM}
        self.my_model = model[opt.model_name](embedding_matrix=self.embedding_matrix,opt=opt).to(opt.device)
        # loss and optimizer
        self.ce_lossFunc = nn.CrossEntropyLoss()                       # 见收藏夹
        params = filter(lambda p: p.requires_grad, self.my_model.parameters())
        self.optimizer = optim.Adam(lr=opt.learn_rate,params=params,weight_decay=opt.l2reg)

    def initial_parameters(self):
        for child in self.my_model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                   #torch.nn.init.uniform_(p, a=-opt.uniform_range,\
                   #                        b=opt.uniform_range)

    def prepare_data(self):
        """
        初始化数据：
            1_生成一个词汇表（包含所有本次实验要用到的word），为每个word找到一个对应的 index
            2_生成 embedding matrix，matrix 每一行代表一个 word 的 vector。对应关系为 matrix
            中word 对应的行的 index 与其在 vocabulary 中的 index 相同。
            3_得到 training、validation、testing 三者的 Dataloader
        :return:
        """
        # 1_构建词汇表
        # 1.1_构建词汇表
        data_file = [opt.train_path,opt.test_path]
        self.vocabulary = get_vocabulary(fnames=data_file,dat_fname = '{0}_tokenizer.dat'.format(opt.dataset),
                                         max_seq_len=opt.max_seq_len)
        # 2_构建 embedding matrix
        self.embedding_matrix = Vectorizer(self.vocabulary,opt.embedding_dim).\
            generate_embeddingMatrix(embedding_fileName=opt.embedding_file_path,\
                                     dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embedding_dim), opt.dataset))
        # 2_构建 Dataset
        # 1) 得到所有数据
        train_val_dataset = My_DataSet(opt.train_path, self.vocabulary,mode=opt.model_name)
        self.test_dataset = My_DataSet(opt.test_path, self.vocabulary,mode=opt.model_name)
        # 2) 划分数据集
        assert 0 < opt.train_ratio <= 1
        train_set_size = int(len(train_val_dataset) * opt.train_ratio)
        val_set_size = len(train_val_dataset) - train_set_size
        if opt.train_ratio < 1:
            self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_set_size, val_set_size])
        else:
            self.train_dataset,self.val_dataset = train_val_dataset,self.test_dataset

    def _train(self):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0                 #代表训练了多少个 mini-batch
        path = None                     #记录最好模型的参数的文件路径
        #整个训练
        for i_epoch in range(opt.epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.my_model.train()             # switch model to training mode
            #更新参数阶段（tarin_dataset）
            for i_batch,batch in enumerate(self.train_dataLoader):
                global_step += 1
                # clear gradient accumulators
                inputs = {k:v.to(opt.device) for k,v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.my_model(inputs)
                targets = inputs['y'].to(opt.device)
                loss = self.ce_lossFunc(outputs,targets)
                loss.backward()
                self.optimizer.step()
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % 10 == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            #选择阶段（validation）
            val_acc, val_f1 = self._evaluate_acc_f1(self.val_dataLoader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(opt.model_name, opt.dataset, round(val_acc, 4))
                torch.save(self.my_model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= opt.patience:
                print('>> early stop.')
                break
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        targets_all, outputs_all = None, None
        # switch model to evaluation mode
        self.my_model.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                inputs = {k: v.to(opt.device) for k, v in batch.items()}
                targets = batch['y'].to(opt.device)
                outputs = self.my_model(inputs)

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
        #构建 Dataloader
        self.train_dataLoader = DataLoader(self.train_dataset, opt.batch_size, shuffle=True)
        self.val_dataLoader = DataLoader(self.val_dataset, opt.batch_size, shuffle=False)
        self.test_dataLoader = DataLoader(self.test_dataset, opt.batch_size, shuffle=False)
        #initial model parameters
        self.initial_parameters()
        #train
        best_model_path = self._train()
        print(best_model_path)
        #test
        self.my_model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(self.test_dataLoader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

def main():
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)            #设置哈希随机值
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    performer = Performer()
    performer.run()

if __name__ == "__main__":
    main()