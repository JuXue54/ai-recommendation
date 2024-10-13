# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import fire
import torch
import torch as t
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torchnet import meter
from torchvision.transforms import ToPILImage
import sys

import models
from config.config import DefaultConfig
from data.dataset import ImageDataPath
from models.RecommendNet import RecommendNet
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


def hello_world():
    print('hello world')

def print_hi(name=None):
    data_path_set = ImageDataPath(opt.train_data_root, train=True)
    train_Loader = DataLoader(data_path_set,
                              batch_size=10,
                              shuffle=True,
                              num_workers=0)
    # for ii,(data,label) in enumerate(train_Loader):
    #     print(label)
    it = enumerate(train_Loader)
    ii, (data, label) = it.__next__()
    to_pil = ToPILImage()
    print(label[0])
    print(data[0].size())
    v = Visualizer()
    v.img('img', data[0])
    pic = to_pil(data[0])
    pic.show()


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


def test():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fire.Fire()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
