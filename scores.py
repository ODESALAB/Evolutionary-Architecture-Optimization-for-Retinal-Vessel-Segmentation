import torch
import numpy as np
from sklearn.metrics import roc_auc_score

#====================================================================
# This part of the code is based on FR-UNet
# from https://github.com/lseventeen/FR-UNet
# Liu, W., Yang, H., Tian, T., Cao, Z., Pan, X., Xu, W., ... & Gao, F. (2022). Full-resolution network and dual-threshold iteration for retinal vessel and coronary angiograph segmentation. IEEE Journal of Biomedical and Health Informatics, 26(9), 4623-4634.
# ===================================================================

def preprocess(predict, target, threshold=0.5):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    
    predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()

    return predict_b, target

def get_confusion_matrix(predict, target):
    predict_b, target = preprocess(predict, target)

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()

    return tp, tn, fp, fn

def auc_score(target, predict):
    predict_b, target = preprocess(predict, target)
    return np.round(roc_auc_score(target, predict_b), 4)

def accuracy_score(tp, tn, fp, fn):
    return np.round((tp + tn) / (tp + fp + fn + tn), 4)

def precision_score(tp, tn, fp, fn):
    return np.round(tp / (tp + fp), 4)

def sensitivity_score(tp, tn, fp, fn):
    return np.round(tp / (tp + fn), 4)

def specificity_score(tp, tn, fp, fn):
    return np.round(tn / (tn + fp), 4)

def iou_score(tp, tn, fp, fn):
    return np.round(tp / (tp + fp + fn), 4)

def f1_score(tp, tn, fp, fn):
    sen = tp / (tp + fn)
    pre = tp / (tp + fp)
    return np.round(2 * pre * sen / (pre + sen), 4)
    
