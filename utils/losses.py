import torch
import torch.nn as nn
from typing import Optional, List
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1. - IoU

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        batch_size = len(inputs)
        scores = []
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)  
        for i in range(batch_size):
            
            #flatten label and prediction tensors
            input = inputs[i].view(-1)
            target = targets[i].view(-1)
            
            intersection = (input * target).sum()                            
            dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
            scores.append(1. - dice)
        return sum(scores) / len(scores)

class CombinedLoss(nn.Module):

    def __init__(self, loss_type = None):
        super(CombinedLoss, self).__init__()
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.jaccard_loss = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(smp.losses.BINARY_MODE)

        self.loss_type = loss_type
    
    def forward(self, output, targets):

        if self.loss_type == 'dice':
            return self.dice_loss(output, targets)
        
        elif self.loss_type == 'bce':
            return self.bce_loss(torch.sigmoid(output), targets)
        
        elif self.loss_type == 'focal':
            return self.focal_loss(torch.sigmoid(output), targets)
        
        elif self.loss_type == 'jaccard':
            return self.jaccard_loss(output, targets)
        
        elif self.loss_type == 'dice + bce':
            dice = self.dice_loss(output, targets)
            bce = self.bce_loss(torch.sigmoid(output), targets)
            return dice + bce
        
        elif self.loss_type == 'dice + jaccard + bce':
            dice = self.dice_loss(output, targets)
            bce = self.bce_loss(torch.sigmoid(output), targets)
            #focal = self.focal_loss(torch.sigmoid(output), targets)
            jaccard = self.jaccard_loss(output, targets)

            return dice + jaccard + bce


        """
        dice = self.dice_loss(output, targets)
        if self.long_runs == False:
            return dice
        else:
            bce = self.bce_loss(torch.sigmoid(output), targets)
            focal = self.focal_loss(torch.sigmoid(output), targets)
            jaccard = self.jaccard_loss(output, targets)

            return dice + jaccard + bce
        """