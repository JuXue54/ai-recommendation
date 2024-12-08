import os

import torch as t
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from db.mappers import StockSnapshotMapper, DatasourceMapper, StockProfitMapper, TradingDayMapper


class ImageDataPath(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        Image data loader
        :param root: root path
        :param transforms:
        :param train:
        :param test:
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


class StockMarketData(data.Dataset):

    def __init__(self, from_date, to_date=None, train=True, test=False):
        """
        Data are so small that all in the memory.
        Adding lazy load in future
        """
        datasource_mapper = DatasourceMapper()
        attributes_mapper = StockSnapshotMapper()
        profit_mapper = StockProfitMapper()
        trading_day_mapper = TradingDayMapper()
        if to_date is None:
            to_date = datasource_mapper.query_date('FUTURE_5_YIELD').date

        all_attribute = attributes_mapper.query_range(from_date, to_date)
        profits = profit_mapper.query_range(from_date, to_date)
        dates = trading_day_mapper.query_range(from_date, to_date)
        stock_codes = list(set(map(lambda p: p.code, profits)))
        stock_num = len(stock_codes)
        if test:
            stock_codes = []
        elif train:
            stock_codes = stock_codes[:int(0.7 * stock_num)]
        else:
            stock_codes = stock_codes[int(0.7 * stock_num):]

        self.size = len(stock_codes)
        self.date_len = len(dates)
        stock_dict = dict([(x, i) for i, x in enumerate(stock_codes)])
        date_dict = dict([(x, i) for i, x in enumerate(dates)])
        self.data = t.Tensor(len(dates), self.size, 28)
        self.label = t.Tensor(len(dates), self.size, 1)
        for x in all_attribute:
            i = date_dict.get(x.date)
            j = stock_dict.get(x.code)
            self.data[i][j] = t.Tensor(
                [x.open, x.close, x.high, x.low, x.buy, x.sell, x.turnover, x.volume, x.bid1, x.bid1_volume,
                 x.bid2, x.bid2_volume, x.bid3, x.bid3_volume, x.bid4, x.bid4_volume, x.bid5, x.bid5_volume,
                 x.ask1, x.ask1_volume, x.ask2, x.ask2_volume, x.ask3, x.ask3_volume,
                 x.ask4, x.ask4_volume, x.ask5, x.ask5_volume])
        for x in profits:
            i = date_dict.get(x.date)
            j = stock_dict.get(x.code)
            self.label[i][j] = t.Tensor([x.yield5])

    def __getitem__(self, index):
        return self.data[:, index, :], self.label[:, index, :]

    def __len__(self):
        return self.size

    def time_len(self):
        return self.date_len
