from torch.utils.data import Dataset
from torch import tensor
import torch
import numpy as np
import pickle
import os

def get_vocabulary(fnames,dat_fname,max_seq_len):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        vocabulary = pickle.load(open(dat_fname, 'rb'))
    else:
        vocabulary = Vocabulary(max_seq_len=max_seq_len)
        for fname in fnames:
            vocabulary.take_into_vocabulary_form_text(text_fileName=fname)
        pickle.dump(vocabulary, open(dat_fname, 'wb'))
    return vocabulary

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Vocabulary(object):
    """
    这个类主要用来构建词汇，使得数据集中的 token 有对应的 index
    """
    def __init__(self,max_seq_len,lower=True):
        self.max_seqLen = max_seq_len
        self.lower = lower
        self._token2idx = {}
        self._idx2token = {}

    def take_into_vocabulary_form_text(self,text_fileName):
        """
        从 text 中分出所有 token 并给其分配 index
        :param text:待处理的文本信息，是个 string 类型，并且形式规定为所有 token 被 ' ' 分开
        """
        text = ""
        file = open(text_fileName,'r',newline='\n',errors='ignore')
        lines = file.readlines()
        file.close()
        for i in range(0,len(lines),3):
            sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
            aspect = lines[i+1].lower().strip()
            text_raw = sen_left + " " + aspect + " " + sen_right
            text += text_raw + " "
        if self.lower:
            text = text.lower()
        tokens = text.split()
        for token in tokens:
            self.add_token(token)

    def add_token(self,token):
        """
        在 vocabulary 加入一个 token
        :param token: 待加入的 token
        :return:返回这个 token 对应的 index
        """
        exist_flag,index = self.lookup_index(token)
        if not exist_flag:
            self._token2idx[token] = index
            self._idx2token[index] = token
        return index

    def lookup_index(self,token):
        if self.lower:
            token = token.lower()
        if token in self._token2idx.keys():
            return True,self._token2idx[token]
        else:
            return False,len(self._token2idx)+1

    def row_to_indexs(self,text_row,reverse = False,padding='post', truncating='post'):
        """
        将一句话转换成一个 index vector
        :param text_row: 一句话（或者是 NLP 中指的 document）
        :param vocabulary: 词汇表（为了找每个 word 对应的 index）
        :return: index vector
        """
        if self.lower:
            text_row = text_row.lower()
        words = text_row.strip().split()
        indexs = [self.lookup_index(word)[1] for word in words]
        if len(indexs) == 0:
            indexs = [0]
        if reverse:
            indexs = indexs[::-1]
        indexs = pad_and_truncate(sequence=indexs,maxlen=self.max_seqLen,padding=padding, truncating=truncating)
        return indexs

    def get_size(self):
        return len(self._token2idx)

    def get_w2i_i2w(self):
        return self._token2idx,self._idx2token

class Vectorizer(object):
    """
    给 Vocabulary 中的 word embed 一个 vector matrix
    """
    def __init__(self,vocabulary,embedding_dim):
        super(Vectorizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._vocabulary = vocabulary
        self._word2idx = vocabulary.get_w2i_i2w()[0]

    def get_embeddingMatrix(self):
        return self._embedding_matrix

    def generate_embeddingMatrix(self,embedding_fileName,dat_fname):
        """
        生成一个 embedding matrix -> 每个行向量代表一个 word 的 embedding，该行向量的下标代表该 word（vocabulary
        中 word2idx中的index）；若没有对应的
        :param embedding_fileName:文件名 -> 该文件中存储形式为：{word:vecter}
        :return: embedding matrix
        """
        if os.path.exists(dat_fname):
            print('loading embedding_matrix:', dat_fname)
            self._embedding_matrix = pickle.load(open(dat_fname, 'rb'))
        else:
            print('loading word vectors...')
            self._embedding_matrix = np.zeros((self._vocabulary.get_size() + 2, self._embedding_dim))
            word_vec = self._load_word_vec(embedding_fileName)
            print('building embedding_matrix:', dat_fname)
            for word, i in self._word2idx.items():
                vec = word_vec.get(word)
                if vec is not None:
                    # words not found in embedding index will be all-zeros.
                    self._embedding_matrix[i] = vec
            pickle.dump(self._embedding_matrix, open(dat_fname, 'wb'))
        return self._embedding_matrix

    def _load_word_vec(self,fname):
        fin = open(fname,'r',encoding='utf-8',newline='\n',errors='ignore')
        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            word, vec = ''.join(tokens[:-self._embedding_dim]), tokens[-self._embedding_dim:]
            if word in self._word2idx.keys():
                word_vec[word] = np.asarray(vec, dtype='float32')
        return word_vec

class My_DataSet(Dataset):
    def __init__(self,data_fileName,vocabulary,mode):
        """
        构建 dataset -> 一个句子用 各个word 在 embedding_matrix 中的下标表示
        例如：" I love you",i、love、you 在 embedding_matrix 对应的下标是 1、2、3，
            那么这句话就可以表示成 [1,2,3]
        :param data_fileName: 存放待预测文本数据的文件名（path)
        :param vocabulary: 用本次要用的 word 构建的 Vocabulary类的对象
        :param mode:选择类型，可选LSTM和TD-LSTM以及TC-LSTM三种，每种的处理方式不同
        """
        self.dataset = []
        file = open(data_fileName,'r',newline='\n',errors='ignore')
        lines = file.readlines()
        file.close()
        if mode == "LSTM":
            for i in range(0,len(lines),3):
                sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
                aspect = lines[i+1].lower().strip()
                polarity = lines[i+2].strip()
                text_row = sen_left + " " + aspect + " " + sen_right
                sample = {'x':vocabulary.row_to_indexs(text_row),
                          'y':tensor(int(polarity)+1,dtype=torch.long)}   # +1 是因为CrossEntropyLoss的targets参数的缘故，
                self.dataset.append(sample)
        elif mode == "TD-LSTM":
            for i in range(0,len(lines),3):
                sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
                aspect = lines[i+1].lower().strip()
                polarity = lines[i+2].strip()
                sen_left_with_aspect = sen_left + " " + aspect
                sen_right_with_aspect = aspect + " " + sen_right
                sample = {'x_left':vocabulary.row_to_indexs(sen_left_with_aspect),
                          'x_right':vocabulary.row_to_indexs(sen_right_with_aspect,reverse=True),
                          'y':tensor(int(polarity)+1,dtype=torch.long)}   # +1 是因为CrossEntropyLoss的targets参数的缘故，
                self.dataset.append(sample)
        elif mode == "TC-LSTM":
            for i in range(0,len(lines),3):
                sen_left,_,sen_right = (s.lower().strip() for s in lines[i].partition("$T$"))
                aspect = lines[i+1].lower().strip()
                polarity = lines[i+2].strip()
                sen_left_with_aspect = sen_left + " " + aspect
                sen_right_with_aspect = aspect + " " + sen_right
                sample = {'x_left':vocabulary.row_to_indexs(sen_left_with_aspect),
                          'x_right':vocabulary.row_to_indexs(sen_right_with_aspect,reverse=True),
                          'aspect':vocabulary.row_to_indexs(aspect),
                          'y':tensor(int(polarity)+1,dtype=torch.long)}   # +1 是因为CrossEntropyLoss的targets参数的缘故，
                self.dataset.append(sample)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

