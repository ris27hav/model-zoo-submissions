import torch
import torch.nn.functional as F
import numpy as np
import argparse

# loss function
def BCEDiceLoss(outputs, labels, smooth=1e-5):
    bce = F.binary_cross_entropy_with_logits(outputs, labels)
    
    outputs = F.sigmoid(outputs)
    batch_size = outputs.shape[0]
    outputs = outputs.view(batch_size, -1)
    labels = labels.view(batch_size, -1)
    intersection = outputs*labels
    
    dice = (2.*intersection.sum(1)+smooth) / (outputs.sum(1)+labels.sum(1)+smooth)
    dice = 1 - dice.sum()/batch_size

    return bce/2 + dice


# metrics
def dice_coef(outputs, labels, smooth=1e-5):
    outputs = torch.sigmoid(outputs).view(-1).data.cpu().numpy()
    labels = labels.view(-1).data.cpu().numpy()
    intersection = (outputs * labels).sum()
    
    dice =  (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
    return dice


def iou(outputs, labels, smooth=1e-5):
    outputs = (torch.sigmoid(outputs).data.cpu().numpy()) > 0.5
    labels = (labels.data.cpu().numpy()) > 0.5
    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    return (intersection + smooth) / (union + smooth)


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')