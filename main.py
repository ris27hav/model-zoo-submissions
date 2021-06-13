import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
import os
import yaml
from collections import OrderedDict

# from dataloader import *
from model import *
from train import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    
    # model config
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=96, type=int)
    parser.add_argument('--input_h', default=96, type=int)
    
    # loss function
    parser.add_argument('--loss', default='BCEDiceLoss')

    # optimizer config
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=3e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    # scheduler config
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    
    # run config
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


config = parse_args()

os.makedirs('models/', exist_ok=True)
with open('models/config.yml', 'w') as f:
    yaml.dump(config, f)


# create model
model = UNetPP(config['input_channels'], config['num_classes'], config['deep_supervision'])
model = model.to(config['device'])

# loss function (criterion)
if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = BCEDiceLoss
criterion = criterion.to(config['device'])

# optimizer
if config['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
else:
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                          nesterov=config['nesterov'], weight_decay=config['weight_decay'])

# scheduler
if config['scheduler'] == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
elif config['scheduler'] == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], 
                                               verbose=1, min_lr=config['min_lr'])
elif config['scheduler'] == 'MultiStepLR':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
else:
    scheduler = None

# load data
train_dl, test_dl = get_loader(config['batch_size'], config['num_workers'])

# training
log = train(config, train_dl, test_dl, model, optimizer, scheduler, criterion, metric=iou)

# analysis
