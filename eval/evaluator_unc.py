################################################
# Evaluator for Uncertainty Estimation Methods
################################################
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from .edl_metrics import compute_confidence_score
from .edl_metrics import compute_confidence_score_categorical


class Uncertainty_Evaluator(object):
    """
    If self.ins_output = False, then we evaluate bag-level prediction performance.
    - In this case, 
        i) the target data['y'] is used for ID confidence evaluation.
        ii) the target data['y_soft'] is used for OOD detection evaluation, where **neg-values** indicate OOD instances.
    
    If self.ins_output = True, then we evaluate instance-level prediction performance.
    - In this case, 
        i) the target data['y'] is used for ID confidence evaluation.
        ii) the target data['y_soft'] is also used for OOD detection evaluation, where **neg. values** indicate OOD instances.
    """
    def __init__(self, **kws):
        super(Uncertainty_Evaluator, self).__init__()
        self.kws = kws

        self.valid_functions = {
            'auc_conf_max_alpha': self._conf_auc_with_max_alpha,
            'aupr_conf_max_alpha': self._conf_aupr_with_max_alpha,
            'mean_conf_max_alpha': self._conf_mean_with_max_alpha,
            'auc_conf_max_prob': self._conf_auc_with_max_prob,
            'aupr_conf_max_prob': self._conf_aupr_with_max_prob,
            'mean_conf_max_prob': self._conf_mean_with_max_prob,
            'auc_conf_alpha0': self._conf_auc_with_alpha0,
            'aupr_conf_alpha0': self._conf_aupr_with_alpha0,
            'mean_conf_alpha0': self._conf_mean_with_alpha0,
            'auc_conf_diff_ent': self._conf_auc_with_diff_ent,
            'aupr_conf_diff_ent': self._conf_aupr_with_diff_ent,
            'mean_conf_diff_ent': self._conf_mean_with_diff_ent,
            'auc_conf_mi': self._conf_auc_with_mi,
            'aupr_conf_mi': self._conf_aupr_with_mi,
            'mean_conf_mi': self._conf_mean_with_mi,
            'auc_conf_exp_ent': self._conf_auc_with_exp_ent, 
            'aupr_conf_exp_ent': self._conf_aupr_with_exp_ent,
            'mean_conf_exp_ent': self._conf_mean_with_exp_ent,
            'auc_det_max_alpha': self._det_auc_with_max_alpha,
            'aupr_det_max_alpha': self._det_aupr_with_max_alpha,
            'mean_det_max_alpha': self._det_mean_with_max_alpha,
            'auc_det_max_prob': self._det_auc_with_max_prob,
            'aupr_det_max_prob': self._det_aupr_with_max_prob,
            'mean_det_max_prob': self._det_mean_with_max_prob,
            'auc_det_alpha0': self._det_auc_with_alpha0,
            'aupr_det_alpha0': self._det_aupr_with_alpha0,
            'mean_det_alpha0': self._det_mean_with_alpha0,
            'auc_det_diff_ent': self._det_auc_with_diff_ent,
            'aupr_det_diff_ent': self._det_aupr_with_diff_ent,
            'mean_det_diff_ent': self._det_mean_with_diff_ent,
            'auc_det_mi': self._det_auc_with_mi,
            'aupr_det_mi': self._det_aupr_with_mi,
            'mean_det_mi': self._det_mean_with_mi,
            'auc_det_exp_ent': self._det_auc_with_exp_ent, 
            'aupr_det_exp_ent': self._det_aupr_with_exp_ent,
            'mean_det_exp_ent': self._det_mean_with_exp_ent,
            'auc_nedet_max_alpha': self._nedet_auc_with_max_alpha,
            'aupr_nedet_max_alpha': self._nedet_aupr_with_max_alpha,
            'auc_nedet_max_prob': self._nedet_auc_with_max_prob,
            'aupr_nedet_max_prob': self._nedet_aupr_with_max_prob,
            'auc_nedet_alpha0': self._nedet_auc_with_alpha0,
            'aupr_nedet_alpha0': self._nedet_aupr_with_alpha0,
            'auc_nedet_diff_ent': self._nedet_auc_with_diff_ent,
            'aupr_nedet_diff_ent': self._nedet_aupr_with_diff_ent,
            'auc_nedet_mi': self._nedet_auc_with_mi,
            'aupr_nedet_mi': self._nedet_aupr_with_mi,
        }
        self.valid_metrics = ['auc_conf_max_alpha', 'aupr_conf_max_alpha', 'mean_conf_max_alpha', 'auc_conf_max_prob', 
            'aupr_conf_max_prob', 'mean_conf_max_prob', 'auc_conf_alpha0', 'aupr_conf_alpha0', 'mean_conf_alpha0', 
            'auc_conf_diff_ent', 'aupr_conf_diff_ent', 'mean_conf_diff_ent', 'auc_conf_mi', 'aupr_conf_mi', 'mean_conf_mi',
            'auc_conf_exp_ent', 'aupr_conf_exp_ent', 'mean_conf_exp_ent', 
            'auc_det_max_alpha', 'aupr_det_max_alpha', 'mean_det_max_alpha', 'auc_det_max_prob', 'aupr_det_max_prob', 
            'mean_det_max_prob', 'auc_det_alpha0', 'aupr_det_alpha0', 'mean_det_alpha0', 'auc_det_diff_ent', 'aupr_det_diff_ent', 
            'mean_det_diff_ent', 'auc_det_mi', 'aupr_det_mi', 'mean_det_mi', 
            'auc_det_exp_ent', 'aupr_det_exp_ent', 'mean_det_exp_ent', 
            'auc_nedet_max_alpha', 'aupr_nedet_max_alpha', 'auc_nedet_max_prob', 'aupr_nedet_max_prob', 'auc_nedet_alpha0', 
            'aupr_nedet_alpha0', 'auc_nedet_diff_ent', 'aupr_nedet_diff_ent', 'auc_nedet_mi', 'aupr_nedet_mi'
        ]

        self.dataset = self.kws['dataset']
        self.ins_output = False
        if 'ins_output' in self.kws and self.kws['ins_output'] == True:
            print("[warning] the input 'y_hat' is the output of instance-level networks.")
            self.ins_output = True

        self.edl_output = False
        if 'edl' in self.kws and self.kws['edl'] == True:
            print("[warning] The input 'y_hat' is the output of EDL networks.")
            self.edl_output = True

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def ret_pred_and_gt(self):
        return {'y_hat': self.y_hat, 'y': self.y, 'y_soft': self.y_soft}

    def _pre_compute(self, data):
        """
        If data['y_hat'] is with a shape of 
          - [N, num_cls], then it is the output of EDL networks or standard classfication networks
          - [N, num_cls * 2], then it is the output of EDL-Beta networks
        """
        self.y, self.y_hat = data['y'], data['y_hat']

        if 'y_soft' in data:
            self.y_soft = data['y_soft']
            if type(self.y_soft) == torch.Tensor:
                self.y_soft = self.y_soft.squeeze().cpu() # [N, ]
            elif type(self.y_soft) == np.ndarray:
                self.y_soft = torch.from_numpy(np.squeeze(self.y_soft)) # [N, ]
        else:
            self.y_soft = None

        if type(self.y) == torch.Tensor:
            self.y = self.y.squeeze().cpu() # [N, ]
        elif type(self.y) == np.ndarray:
            self.y = torch.from_numpy(np.squeeze(self.y)) # [N, ]
        
        if type(self.y_hat) == torch.Tensor:
            self.y_hat = self.y_hat.squeeze().cpu() # [N, x]
        elif type(self.y_hat) == np.ndarray:
            self.y_hat = torch.from_numpy(np.squeeze(self.y_hat)) # [N, x]
        
        assert self.y.shape[0] == self.y_hat.shape[0]

        # ONLY FOR standard clf models 
        # we check the input dims and convert them to standard prob outputs, [N, num_cls], sum_p_i = 1
        if not self.edl_output:
            if self.ins_output:
                # for instance outputs, we assume they are probabilities of positive class.
                assert len(self.y_hat.shape) == 1
                self.y_hat = torch.cat([(1 - self.y_hat).unsqueeze(-1), self.y_hat.unsqueeze(-1)], dim=1)
            else:
                # for bag outputs, we assume they are logits outputs 
                assert len(self.y_hat.shape) == 2
                self.y_hat = F.softmax(self.y_hat, dim=1)
        
        # filter predictions with np.inf or np.nan
        sel_idx = ~(torch.isinf(self.y_hat).any(dim=-1) | torch.isnan(self.y_hat).any(dim=-1))
        sel_idx &= ~(torch.isinf(self.y) | torch.isnan(self.y))
        self.y = self.y[sel_idx]
        self.y_hat = self.y_hat[sel_idx]
        if self.y_soft is not None:
            self.y_soft = self.y_soft[sel_idx]

    def _uncertainty_evaluation(self, task, uncertainty_type, ret_metrics, ret_scores=False, use_neg_output=False):
        if task == 'IDConf':
            # Confidence evaluation for ID samples
            # use 'y_soft' to filter OOD samples and only keep ID samples 
            Y = self.y[self.y_soft >= 0]
            Y_hat = self.y_hat[self.y_soft >= 0]
            corrects = (Y == Y_hat.max(-1)[-1]).numpy()
        elif task == 'OODDet':
            # OOD detection evaluation for ID / OOD samples
            Y = (self.y_soft >= 0).to(torch.long)
            Y_hat = self.y_hat
            corrects = Y
        elif task == 'NeOODDet':
            # OOD detection evaluation for ID / OOD Negative samples
            Y = (self.y_soft >= 0).to(torch.long)
            Y_hat = self.y_hat[self.y <= 0]
            corrects = Y[self.y <= 0]
        else:
            pass
        
        if self.edl_output:
            scores = compute_confidence_score(Y_hat, uncertainty_type)
        else:
            scores = compute_confidence_score_categorical(Y_hat, uncertainty_type)

        # when there is no OOD sample
        if (corrects == 1).all() or (corrects == 0).all():
            auroc, aupr = 0, 0
        else:
            if scores is not None:
                # filter predictions with np.inf or np.nan
                sel_idx = ~(np.isinf(scores) | np.isnan(scores))
                scores = scores[sel_idx]
                corrects = corrects[sel_idx]
                fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
                auroc = metrics.auc(fpr, tpr)
                aupr = metrics.average_precision_score(corrects, scores)
            else:
                auroc, aupr = 0, 0

        ret = []
        ret_metrics = ret_metrics if isinstance(ret_metrics, list) else [ret_metrics]
        for m in ret_metrics:
            if m == 'auc':
                ret.append(auroc)
            elif m == 'aupr':
                ret.append(aupr)
            elif m == 'mean':
                if scores is None:
                    mean = 0
                else:
                    if task == 'IDConf':
                        mean = scores.mean() # all ID samples
                    elif task == 'OODDet' or task == 'NeOODDet':
                        sel_scores = scores[corrects == 0] # OOD samples
                        mean = 0 if len(sel_scores) == 0 else sel_scores.mean()
                    else:
                        mean = 0
                ret.append(mean)
            elif m == 'score':
                ret.append(scores)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
    def _conf_auc_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_alpha', 'auc', **kws)

    def _conf_aupr_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_alpha', 'aupr', **kws)
    
    def _conf_mean_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_alpha', 'mean', **kws)
    
    def _conf_auc_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_prob', 'auc', **kws)

    def _conf_aupr_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_prob', 'aupr', **kws)
    
    def _conf_mean_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'max_prob', 'mean', **kws)

    def _conf_auc_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'alpha0', 'auc', **kws)

    def _conf_aupr_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'alpha0', 'aupr', **kws)
    
    def _conf_mean_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'alpha0', 'mean', **kws)

    def _conf_auc_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'differential_entropy', 'auc', **kws)

    def _conf_aupr_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'differential_entropy', 'aupr', **kws)
    
    def _conf_mean_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'differential_entropy', 'mean', **kws)
    
    def _conf_auc_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'expected_entropy', 'auc', **kws)

    def _conf_aupr_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'expected_entropy', 'aupr', **kws)
    
    def _conf_mean_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'expected_entropy', 'mean', **kws)

    def _conf_auc_with_mi(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'mutual_information', 'auc', **kws)

    def _conf_aupr_with_mi(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'mutual_information', 'aupr', **kws)
    
    def _conf_mean_with_mi(self, **kws):
        return self._uncertainty_evaluation('IDConf', 'mutual_information', 'mean', **kws)
    
    def _det_auc_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_alpha', 'auc', **kws)

    def _det_aupr_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_alpha', 'aupr', **kws)
    
    def _det_mean_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_alpha', 'mean', **kws)
    
    def _det_auc_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_prob', 'auc', **kws)

    def _det_aupr_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_prob', 'aupr', **kws)
    
    def _det_mean_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'max_prob', 'mean', **kws)

    def _det_auc_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'alpha0', 'auc', **kws)

    def _det_aupr_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'alpha0', 'aupr', **kws)
    
    def _det_mean_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'alpha0', 'mean', **kws)
    
    def _det_auc_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'expected_entropy', 'auc', **kws)

    def _det_aupr_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'expected_entropy', 'aupr', **kws)
    
    def _det_mean_with_exp_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'expected_entropy', 'mean', **kws)

    def _det_auc_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'differential_entropy', 'auc', **kws)

    def _det_aupr_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'differential_entropy', 'aupr', **kws)
    
    def _det_mean_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'differential_entropy', 'mean', **kws)

    def _det_auc_with_mi(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'mutual_information', 'auc', **kws)

    def _det_aupr_with_mi(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'mutual_information', 'aupr', **kws)
    
    def _det_mean_with_mi(self, **kws):
        return self._uncertainty_evaluation('OODDet', 'mutual_information', 'mean', **kws)
    
    def _nedet_auc_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'max_alpha', 'auc', **kws)

    def _nedet_aupr_with_max_alpha(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'max_alpha', 'aupr', **kws)
    
    def _nedet_auc_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'max_prob', 'auc', **kws)

    def _nedet_aupr_with_max_prob(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'max_prob', 'aupr', **kws)

    def _nedet_auc_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'alpha0', 'auc', **kws)

    def _nedet_aupr_with_alpha0(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'alpha0', 'aupr', **kws)

    def _nedet_auc_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'differential_entropy', 'auc', **kws)

    def _nedet_aupr_with_diff_ent(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'differential_entropy', 'aupr', **kws)

    def _nedet_auc_with_mi(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'mutual_information', 'auc', **kws)

    def _nedet_aupr_with_mi(self, **kws):
        return self._uncertainty_evaluation('NeOODDet', 'mutual_information', 'aupr', **kws)

    def compute(self, data):
        self._pre_compute(data)
        res_metrics = dict()
        metrics = self.valid_metrics
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()
        return res_metrics
        