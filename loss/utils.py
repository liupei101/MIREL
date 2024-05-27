import torch
import torch.nn as nn

from .loss_clf import BinaryCrossEntropy, SoftTargetCrossEntropy
from .loss_edl import get_edl_loss, edl_neg_instance_loss

def load_loss(task, *args, **kws):
    if task == 'clf':
        if 'edl' in kws and kws['edl']:
            loss_fn = get_edl_loss(kws['edl_type'])
        else:
            loss_fn = SoftTargetCrossEntropy(kws['smoothing'])
        return loss_fn
    else:
        pass
        return None

def load_edl_neg_instance_loss():
    return edl_neg_instance_loss

def loss_reg_l1(coef):
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func
