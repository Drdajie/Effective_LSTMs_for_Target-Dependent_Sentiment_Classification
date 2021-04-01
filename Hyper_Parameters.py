import torch
import logging
import sys

#将类当作命名空间用
class opt:
    #file_args:
    dataset = 'twitter'
    data = {
            'twitter':
                {'train_path':"../../Dataset/twitter/train.raw",
                 'test_path' : "../../Dataset/twitter/test.raw"},
            'restaurant':
                {'train_path':"../Dataset/semeval14/Restaurants_Train.xml.seg",
                 'test_path':"../Dataset/semeval14/Restaurants_Test_Gold.xml.seg"},
            'laptop':
                {'train_path':"../Dataset/semeval14/Laptops_Train.xml.seg",
                 'test_path':"../Dataset/semeval14/Laptops_Test_Gold.xml.seg"}
            }
    train_path,test_path = data[dataset].values() # the path of traning data & the path of test data
    embedding_name = '100_27'
    embedding_set = {
            '100_27':"./Data/glove.twitter.27B.100d.txt",
            '300_84':"../Word_Embedding/Glove/glove.840B.300d.txt",
            '300_42':""}
    embedding_file_path = embedding_set[embedding_name]

    #preTrain_args:                #主要的超参数
    train_ratio = 1            #the size ratio of train
    validation_ratio = 0  #the size ratio of validation
    embedding_dim = 100     #dimension of word embedding
    max_seq_len = 80
    paded_mark = 0
    seed = int(1234)
    uniform_range = 0.003
    l2reg = 3e-4

    #model_args:
    model_name = 'TC-LSTM'#'TD-LSTM'        #'LSTM'
    input_dim = embedding_dim
    hidden_dim = 300
    num_layer = 1
    bias = True
    batch_first = True
    num_class = 3

    #train_args:
    epoch = 30
    learn_rate = 0.01       #learning rate
    batch_size = 16         #the mini-batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 5

#预处理
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))