# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import sys
import time

import fire
import torch
import torch as t
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torchnet import meter

import models
from config.config import DefaultConfig
from core.transformers import FinancialDataTransformer
from data.dataset import ImageDataPath, StockMarketData
from db.mappers import StockSnapshotMapper
from source.easyQuotationSource import EasyQuotationSource
from utils.Visualizer import Visualizer

try:
    import ipdb
except:
    import pdb as ipdb

opt = DefaultConfig()

if opt.use_gpu and sys.platform.startswith('darwin'):
    mps_device = t.device("mps")
else:
    mps_device = None


def to_gpu(data):
    if opt.use_gpu:
        if sys.platform.startswith('darwin'):
            return data.to(mps_device)
        elif sys.platform.startswith('linux'):
            return data.cuda()
        else:
            raise Exception('Unsupported os')
    return data


def test_db():
    transformer = FinancialDataTransformer()
    transformer.sync_all_future_yield()
    # print(datetime.date.today())
    # transformer.sync_trading_day()


def snapshot():
    curr = datetime.datetime.now()
    print(f"start to snapshot")
    source = EasyQuotationSource()
    data = source.snapshot()

    def transform_to_snapshot(items):
        k, v = items
        v['code'] = k
        v.pop('time')
        return v

    res = list(map(transform_to_snapshot, data.items()))
    rows = StockSnapshotMapper().insert_or_update_batch(res)
    curr_1 = datetime.datetime.now()
    # synchronize trading day
    print(f"End snapshot and start sync trading day, time cost: {curr_1 - curr}")
    transformer = FinancialDataTransformer()
    transformer.sync_trading_day()
    # synchronize future yield
    curr_2 = datetime.datetime.now()
    print(f"End sync trading day and start sync future yield, time cost: {curr_2 - curr_1}")
    transformer.sync_all_future_yield()
    curr_3 = datetime.datetime.now()
    print(f"End sync future yield, time cost: {curr_3 - curr_2}")
    print(f'current time is {curr_3}, return row count are {rows}')


def try_visualizer():
    v = Visualizer()
    v.plot('loss', 1)
    v.plot('loss', 0.8)
    v.plot('loss', 0.6)


def train(**kwargs):
    vis = Visualizer(opt.env)
    # step1: Model
    model = getattr(models, opt.model)(num_classes=2, dropout=0.5, name=opt.model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    to_gpu(model)

    # step2: data
    train_data_path = ImageDataPath(opt.train_data_root, train=True)
    val_data_path = ImageDataPath(opt.train_data_root, train=False)
    train_data_loader = DataLoader(train_data_path, opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.num_workers)
    val_data_loader = DataLoader(val_data_path, opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    # step3: object function and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)
    # step4: statistic factor
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_data_loader):
            input = V(data)
            target = V(label)
            if opt.use_gpu:
                input = to_gpu(input)
                target = to_gpu(target)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            loss_meter.add(loss.data.item())
            confusion_matrix.add(score.data, target.data)

            # if ii % opt.print_freq == opt.print_freq - 1:
            vis.plot('loss', loss_meter.value()[0])
            # print('Train: The first target is %s, predicted value is %s' % (target[0], score[0]))

        if opt.need_save:
            model.save()

        val_cm, val_accuracy = val(model, val_data_loader, opt)
        vis.plot('val_accuracy', val_accuracy)
        vis.log('epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}'
                .format(epoch=epoch,
                        loss=loss_meter.value()[0],
                        val_cm=str(val_cm.value()),
                        train_cm=str(confusion_matrix.value()),
                        lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader, opt):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    with torch.no_grad():
        for ii, (input, label) in enumerate(dataloader):
            val_input = V(input)
            val_input = to_gpu(val_input)
            score = model(val_input)
            confusion_matrix.add(score.data, label)
            # print('Validate: The first target is %s, predicted value is %s' % (label[0], score.data[0]))
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def collate_fn(batch):
    # print(batch)
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs).permute(1, 0, 2)
    labels = torch.stack(labels).permute(1, 0, 2)
    return inputs, labels


def test():
    vis = Visualizer(opt.env)
    # step1: Model
    model = getattr(models, opt.model)(input_dim=opt.attribute_num, hidden_dim=1, target_dim=1,
                                       batch_size=opt.batch_size,
                                       name=opt.model, device_func=to_gpu)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    to_gpu(model)
    # step2: data
    data = StockMarketData(from_date=datetime.date(2024, 9, 1))

    train_data_loader = DataLoader(data, opt.batch_size,
                                   shuffle=True,
                                   num_workers=opt.num_workers,
                                   collate_fn=collate_fn,
                                   drop_last=True)
    # for ii, (data, label) in enumerate(train_data_loader):
    #     print(data.size())
    #     print(label.size())

    # step3: object function and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)
    # step4: statistic factor
    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii, (data, label) in enumerate(train_data_loader):
            input = V(data)
            target = V(label)
            if opt.use_gpu:
                input = to_gpu(input)
                target = to_gpu(target)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            loss_meter.add(loss.data.item())
            vis.plot('loss', loss_meter.value()[0])

        if opt.need_save:
            model.save()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fire.Fire()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
