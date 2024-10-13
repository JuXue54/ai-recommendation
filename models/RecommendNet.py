import torch
from torch import nn

from models.BasicModule import BasicModule


class RecommendNet(BasicModule):

    def __init__(self, input_dim, hidden_dim, target_dim, batch_size:int =1, layers:int=1, name:str= 'RecommendNet'):
        super(RecommendNet, self).__init__(name)
        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.features = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, layers),
            nn.Linear(hidden_dim, target_dim)
        )
        self.hidden = self.init_hidden(hidden_dim, batch_size, layers)

    def init_hidden(self, hidden_dim, batch_size, layers):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(layers, batch_size, hidden_dim),
                torch.zeros(layers, batch_size, hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.features(input, self.hidden)
        res = nn.Softmax(lstm_out)
        return res