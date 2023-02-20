import os
import copy
import math
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import accuracy, iou_score, sensitivity, specificity, positive_predictive_value, f1_score, recall
from sklearn.metrics import rand_score, roc_auc_score, roc_curve, auc

def auc_score(y_true, y_pred):
    ground_truth_labels = y_true.astype('int').ravel() # we want to make them into vectors
    score_value = 1-y_pred.ravel()/255.0 # we want to make them into vectors
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
    return roc_auc

def iou_frnet(tp, fp, fn, tn):
    return (tp) / (tp + fp + fn)

def f1_score_frnet(tp, fp, fn, tn):
    return (2*tp) / (2*tp + fp + fn)

def sensibility(tp, fp, fn, tn):
    return 1 - fp / (tp + fn)

def conformity(tp, fp, fn, tn):
    return 1 - (fp + fn) / tp

def global_consistency_error(tp, fp, fn, tn):
    """ Lower is better """
    if (tp + fn) == 0 or (tn + fp) == 0 or (tp + fp) == 0 or (tn + fn) == 0:
        return -1

    n = tp + tn + fp + fn
    e1 = (fn * (fn + 2 * tp) / (tp + fn) + fp * (fp + 2 * tn) / (tn + fp)) / n
    e2 = (fp * (fp + 2 * tp) / (tp + fp) + fn * (fn + 2 * tn) / (tn + fn)) / n

    return min(e1, e2)

def rand_index(y_true, y_pred):
    """
        İkisi de aynı sonucu döndürüyor.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html
        https://github.com/rundherum/pymia/blob/master/pymia/evaluation/metric/categorical.py
    """
    return rand_score(y_true.astype(int).ravel(), y_pred.astype(int).ravel())

def adjusted_rand_index(tp, fp, fn, tn):
    n = tn + fp + fn + tp

    fp_tn = tn + fp
    tp_fn = fn + tp
    tn_fn = tn + fn
    tp_fp = fp + tp
    nis = tn_fn * tn_fn + tp_fp * tp_fp
    njs = fp_tn * fp_tn + tp_fn * tp_fn
    sum_of_squares = tp * tp + tn * tn + fp * fp + fn * fn

    a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2.
    b = (njs - sum_of_squares) / 2.
    c = (nis - sum_of_squares) / 2.
    d = (n * n + sum_of_squares - nis - njs) / 2.

    x1 = a - ((a + c) * (a + b) / (a + b + c + d))
    x2 = ((a + c) + (a + b)) / 2.
    x3 = ((a + c) * (a + b)) / (a + b + c + d)
    denominator = x2 - x3

    if denominator != 0:
        return x1 / denominator
    else:
        return 0

def mutual_information(tp, fp, fn, tn):
    n = tn + fp + fn + tp

    fn_tp = fn + tp
    fp_tp = fp + tp

    if fn_tp == 0 or fn_tp / n == 1 or fp_tp == 0 or fp_tp / n == 1:
        return -1

    h1 = -((fn_tp / n) * math.log2(fn_tp / n) + (1 - fn_tp / n) * math.log2(1 - fn_tp / n))
    h2 = -((fp_tp / n) * math.log2(fp_tp / n) + (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

    p00 = 1 if tn == 0 else (tn / n)
    p01 = 1 if fn == 0 else (fn / n)
    p10 = 1 if fp == 0 else (fp / n)
    p11 = 1 if tp == 0 else (tp / n)

    h12 = -((tn / n) * math.log2(p00) +
            (fn / n) * math.log2(p01) +
            (fp / n) * math.log2(p10) +
            (tp / n) * math.log2(p11))

    mi = h1 + h2 - h12
    return mi

def variation_of_information(tp, fp, fn, tn):
    n = tn + fp + fn + tp
    fn_tp = fn + tp
    fp_tp = fp + tp

    if fn_tp == 0 or fn_tp / n == 1 or fp_tp == 0 or fp_tp / n == 1:
        return -1

    h1 = -((fn_tp / n) * math.log2(fn_tp / n) + (1 - fn_tp / n) * math.log2(1 - fn_tp / n))
    h2 = -((fp_tp / n) * math.log2(fp_tp / n) + (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

    p00 = 1 if tn == 0 else (tn / n)
    p01 = 1 if fn == 0 else (fn / n)
    p10 = 1 if fp == 0 else (fp / n)
    p11 = 1 if tp == 0 else (tp / n)

    h12 = -((tn / n) * math.log2(p00) +
            (fn / n) * math.log2(p01) +
            (fp / n) * math.log2(p10) +
            (tp / n) * math.log2(p11))

    mi = h1 + h2 - h12

    vi = h1 + h2 - 2 * mi
    return vi

def interclass_correlation(y_true, y_pred):
    gt = y_true
    seg = y_pred

    n = len(gt)
    mean_gt = gt.mean()
    mean_seg = seg.mean()
    mean = (mean_gt + mean_seg) / 2

    m = (gt + seg) / 2
    ssw = np.power(gt - m, 2).sum() + np.power(seg - m, 2).sum()
    ssb = np.power(m - mean, 2).sum()

    ssw /= n
    ssb = ssb / (n - 1) * 2

    if (ssb + ssw) == 0:
        return -1

    return (ssb - ssw) / (ssb + ssw)

def probabilistic_distance(y_true, y_pred):
    """
        Lower is better??
    """
    gt = y_true.astype(np.int8)
    seg = y_pred.astype(np.int8)

    probability_difference = np.absolute(gt - seg).sum()
    probability_joint = (gt * seg).sum()

    if probability_joint != 0:
        return probability_difference / (2. * probability_joint)
    else:
        return -1

def cohens_cappa(tp, fp, fn, tn):
    agreement = tp + tn
    chance0 = (tn + fn) * (tn + fp)
    chance1 = (fp + tp) * (fn + tp)
    sum_ = tn + fn + fp + tp
    chance = (chance0 + chance1) / sum_

    if (sum_ - chance) == 0:
        return -1

    return (agreement - chance) / (sum_ - chance)

def hausdorff_distance_95th_quantile(y_true, y_pred):
    return hd95(y_true, y_pred)

def average_hausdorff_distance(y_true, y_pred):
    img_pred = sitk.GetImageFromArray(y_pred)
    #img_pred.SetSpacing(self.spacing[::-1])
    img_ref = sitk.GetImageFromArray(y_true)
    #img_ref.SetSpacing(self.spacing[::-1])

    distance_filter = sitk.HausdorffDistanceImageFilter()
    distance_filter.Execute(img_pred, img_ref)
    return distance_filter.GetAverageHausdorffDistance()

def mahalanobis_distance(y_true, y_pred):
    gt_n = np.count_nonzero(y_true)
    seg_n = np.count_nonzero(y_pred)

    if gt_n == 0:
        return -1
    if seg_n == 0:
        return -1

    gt_indices = np.flip(np.where(y_true == 1), axis=0)
    gt_mean = gt_indices.mean(axis=1)
    gt_cov = np.cov(gt_indices)

    seg_indices = np.flip(np.where(y_pred == 1), axis=0)
    seg_mean = seg_indices.mean(axis=1)
    seg_cov = np.cov(seg_indices)

    # calculate common covariance matrix
    common_cov = (gt_n * gt_cov + seg_n * seg_cov) / (gt_n + seg_n)
    common_cov_inv = np.linalg.inv(common_cov)

    mean = gt_mean - seg_mean
    return math.sqrt(mean.dot(common_cov_inv).dot(mean.T))

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true.ravel().astype(int), y_pred.ravel().astype(int))