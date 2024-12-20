import torch
from torch import nn

from models.BasicModule import BasicModule


class RecommendNet(BasicModule):

    def __init__(self, input_dim, hidden_dim, target_dim, batch_size: int = 1, layers: int = 1,
                 name: str = 'RecommendNet', device_func=lambda x: x):
        super(RecommendNet, self).__init__(name)
        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers)
        # self.linear = nn.Linear(hidden_dim, target_dim)
        # self.main = nn.Sequential(
        #     self.lstm,
        #     nn.Flatten(),
        #     self.linear
        # )
        self.hidden = self.init_hidden(hidden_dim, batch_size, layers, device_func)

    def init_hidden(self, hidden_dim, batch_size, layers, device_func):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (device_func(torch.zeros(layers, batch_size, hidden_dim)),
                device_func(torch.zeros(layers, batch_size, hidden_dim)))

    def forward(self, input):
        res, _ = self.lstm(input, self.hidden)
        # res = self.linear(res[:,-1,:])
        # res = nn.Softmax(res)
        return res