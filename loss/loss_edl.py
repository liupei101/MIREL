import torch
from functools import partial

from utils.func import get_device, one_hot_encode

EPS = 1e-16

def aggregate_instance_evidence(instance_evidence, method, weight=None):
    """
    Function used for aggregating instance evidence for instance-level weak supervision.
    """
    # input is with shape of [1, N, num_cls]
    if method == 'mean':
        mean_evidence = torch.sum(instance_evidence, 1) / instance_evidence.shape[1] # [1, num_cls]
        return mean_evidence + 1
    elif method == 'max':
        # select the instance with the largest positive prob.
        temp_alpha = instance_evidence.detach() + 1
        temp_prob = temp_alpha[:, :, [1]] / torch.sum(temp_alpha, dim=-1, keepdim=True) # [1, N, 1]
        sel_idx = torch.argmax(temp_prob.detach())
        agg_evidence = instance_evidence[:, sel_idx, :] # [1, num_cls]
        return agg_evidence + 1 
    elif method == 'diweight':
        assert weight.shape[1] == instance_evidence.shape[1]
        if len(weight.shape) > 2 and weight.shape[-1] > 1:
            assert weight.shape[-1] == instance_evidence.shape[-1]
            w = weight / torch.sum(weight, -1, keepdim=True) # [1, N, 2], instance-level expectation
            w = w / torch.sum(w, 1, keepdim=True) # [1, N, 2], column-normalized probs
            cur_weight = w[:, :, [1]] # [1, N, 1], normalized positive probs 
        else:
            cur_weight = weight

        weighted_evidence = torch.sum(cur_weight * instance_evidence, dim=1) # [1, num_cls]
        return weighted_evidence + 1
    else:
        raise NotImplementedError("{} cannot be recognized.".format(method))

def compute_auxiliary_output(ins_alpha, target, separate='II', aggregate='diweight', weight=None):
    """
    Input:
        ins_alpha: the instance-level output of mini-batch, should be a list of Tensor [1, N, num_cls].
        target: one-hot label of mini-batch, should be a Tensor with shape of [B, num_cls].
        separate: compute the loss sepatately for neg. and pos. bags.
        aggregate: 'mean', 'max', 'diweight' for instance evidence aggregation.
        weight: it is used to aggregate instance evidence if it is not None and aggregate='diweight'.

    Output:
        pos_/neg_alpha: bag-level output, if target is positive; otherwise, it is instance-level output.
        pos_/neg_target: bag-level target, if positive; otherwise, instance-level target.
    """
    # pos & neg: compute seperately
    if separate == "II":
        num_classes = target.shape[-1]
        neg_idx = torch.nonzero(target[:, 0] == 1).squeeze(-1).tolist()
        pos_idx = torch.nonzero(target[:, 1] == 1).squeeze(-1).tolist()
        # For negative bags, we directly use instance-level prediction
        if len(neg_idx) == 0:
            neg_alpha, neg_target = None, None
        else:
            neg_alpha = [ins_alpha[i].squeeze(0) for i in neg_idx] # list of Tensor [N, num_cls]
            neg_target = [torch.zeros(x.shape[0], dtype=target.dtype, device=target.device) for x in neg_alpha]
            neg_target = [one_hot_encode(x, num_classes, dtype=target.dtype, device=target.device) for x in neg_target]
        # For positive bags, we use aggregated prediction
        if len(pos_idx) == 0:
            pos_alpha, pos_target = None, None
        else:
            pos_target = target[pos_idx] # [B_pos, num_cls]
            pos_alpha = []
            for i in pos_idx:
                ins_evidence = ins_alpha[i] - 1
                w = weight[i] if weight is not None else None
                bag_alpha = aggregate_instance_evidence(ins_evidence, aggregate, weight=w) # [1, num_cls]
                pos_alpha.append(bag_alpha)
            pos_alpha = torch.cat(pos_alpha, dim=0) # [B_pos, num_cls]
        return pos_alpha, pos_target, neg_alpha, neg_target
    
    # pos & neg: compute in a same way
    elif separate == "I":
        n_batch, alpha = len(ins_alpha), []
        for i in range(n_batch):
            ins_evidence = ins_alpha[i] - 1
            w = weight[i] if weight is not None else None
            bag_alpha = aggregate_instance_evidence(ins_evidence, aggregate, weight=w) # [1, num_cls]
            alpha.append(bag_alpha)
        alpha = torch.cat(alpha, dim=0) # [B, num_cls]
        return alpha, target
    
    else:
        pass

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device) 
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term # [B, 1]
    return kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = alpha.device
    assert y.shape[-1] != 1, "y must be one-hot encoded."
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var # [B, 1]
    return loglikelihood

def loglikelihood_loss_with_fisher(y, alpha, device=None):
    if not device:
        device = alpha.device
    assert y.shape[-1] != 1, "y must be one-hot encoded."
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    gamma1_alpha = torch.polygamma(1, alpha)
    gamma1_S = torch.polygamma(1, S)

    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2 * gamma1_alpha, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        gamma1_alpha * alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var

    loglikelihood_det_fisher = - (
        torch.log(gamma1_alpha).sum(-1, keepdim=True) + 
        torch.log(1.0 - (gamma1_S / gamma1_alpha).sum(-1, keepdim=True))
    )

    return loglikelihood, loglikelihood_det_fisher

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device=None):
    if not device:
        device = alpha.device
    if y.shape[-1] == 1:
        y = one_hot_encode(y, num_classes, device, alpha.dtype)
    y = y.to(device)
    alpha = alpha.to(device)

    if c_fisher is None or c_fisher < 0:
        loglikelihood = loglikelihood_loss(y, alpha, device=device)
    else:
        loglikelihood, loglikelihood_det_fisher = loglikelihood_loss_with_fisher(y, alpha, device=device)
        if c_fisher > 1e-5:
            loglikelihood = loglikelihood + c_fisher * loglikelihood_det_fisher

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    if use_kl_div:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    else:
        kl_div = 0
    
    return loglikelihood + kl_div # [B, 1]

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, use_kl_div, device=None):
    if not device:
        device = alpha.device
    if y.shape[-1] == 1:
        y = one_hot_encode(y, num_classes, device, alpha.dtype)
    y = y.to(device)
    alpha = alpha.to(device) # [B, num_cls]

    S = torch.sum(alpha, dim=1, keepdim=True) # [B, 1]
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True) # [B, 1]

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    if use_kl_div:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    else:
        kl_div = 0
    # print("EDL loss:", A, "its mean:", A.mean())
    # print("KL Div:", kl_div, "its mean:", kl_div.mean())

    return A + kl_div

# the loss to avoid zero-evidence region for EDL models (pandey et al, ICML, 2023)
def edl_red_loss(target, output_alpha, num_classes, loss_type='log-alpha', device=None):
    if not device:
        device = output_alpha.device
    if target.shape[-1] == 1:
        target = one_hot_encode(target, num_classes, device, output_alpha.dtype)
    
    target = target.to(device) # [B, num_cls]
    output_alpha = output_alpha.to(device) # [B, num_cls]
    S = torch.sum(output_alpha, dim=1, keepdim=True) # [B, 1]
    cor = num_classes / S # [B, 1]
    
    # instance pred: [alpha, beta]
    target_alpha = (target * output_alpha).sum(-1, keepdim=True) # [B, 1]
    if loss_type == 'log-alpha':
        edv_target = target_alpha - 1
        edv_target += EPS
        loss = -1 * cor * torch.log(edv_target) # pandey, ICML, 2023
    elif loss_type == 'custom':
        edv_target = target_alpha - 1
        edv_target += EPS
        loss = - 1 / edv_target  * torch.log(edv_target) # custom
    elif loss_type == 'expec-alpha':
        loss = -1 * cor * torch.log(target_alpha / S) # expectation nll loss
    elif loss_type == 'bayes-alpha-digamma':
        loss = -1 * cor * (torch.digamma(target_alpha) - torch.digamma(S)) # Bayes risk
    elif loss_type == 'bayes-alpha-log':
        loss = -1 * cor * (torch.log(target_alpha) - torch.log(S)) # Bayes risk
    else:
        loss = None
    # print("RED loss:", loss, "its mean:", loss.mean())
    
    return loss # [B, 1]

def edl_mse_loss(output_alpha, target, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, red_type='log-alpha', 
    branch='bag', separate='II', aggregate='mean', weight=None, ins_target=None, device=None):
    if not device:
        if isinstance(output_alpha, list):
            device = output_alpha[0].device
            dtype  = output_alpha[0].dtype
        else:
            device = output_alpha.device
            dtype  = output_alpha.dtype
    if target.shape[-1] == 1:
        target = one_hot_encode(target, num_classes, device, dtype)
    if branch == 'bag':
        if red_type is not None and len(red_type) > 0:
            loss = torch.mean(
                mse_loss(target, output_alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device) + 
                edl_red_loss(target, output_alpha, num_classes, red_type, device)
            )
        else:
            loss = torch.mean(
                mse_loss(target, output_alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device)
            )
    elif branch == 'instance':
        assert isinstance(output_alpha, list), "Output should be a list of each sample's instance-level alpha."
        
        if separate in ['F', 'B']:
            n_sample = len(output_alpha)
            all_alpha = [output_alpha[i].squeeze(0) for i in range(n_sample)] # obtain a list of Tensor [N, num_cls]
            num_cls_ins = all_alpha[0].shape[-1] # the dim of instance prediction
            
            # get instance labels
            # 1: a fully-supervised baseline, ONLY used as a FS model for the comparison to WS counterparts
            # `ins_target` is only used when separate = 'F'.
            if separate == 'F': 
                assert isinstance(ins_target, list), "`ins_target` should a list of instance labels."
                assert n_sample == len(ins_target)
                all_target = [ins_target[i].squeeze(0).cuda() for i in range(n_sample)]
                all_target = [one_hot_encode(x, num_cls_ins, dtype=target.dtype, device=target.device) for x in all_target]
            # 2: a baseline in which instances directly inhert labels from their parent bag
            else: 
                assert n_sample == len(target)
                all_target = [target[[i], :] * torch.ones((all_alpha[i].shape[0], num_cls_ins), dtype=target.dtype, device=target.device) 
                                for i in range(n_sample)] # duplicate bag label as instances' label
            
            ins_loss = 0
            for i in range(n_sample):
                # applying weighted loss to the instances of positive bags
                num_ins = all_alpha[i].shape[0]
                if aggregate == 'diweight' and target[i, 1].item() == 1: 
                    w = weight[i].squeeze(0) # [N, num_cls]
                    assert w.shape == all_alpha[i].shape
                    prob = w / torch.sum(w, -1, keepdim=True) # [N, num_cls], instance-level expectation
                    norm_prob = prob / torch.sum(prob, 0, keepdim=True) # [N, num_cls], column-normalized probs
                    ins_weight = norm_prob[:, [1]] * num_ins # [N, 1], weight = N * normalized positive probs 
                else:
                    ins_weight = torch.ones((all_alpha[i].shape[0], 1)).cuda() # [N, 1], means no weight

                if red_type is not None and len(red_type) > 0:
                    ins_loss += torch.mean(
                        ins_weight * (mse_loss(all_target[i], all_alpha[i], epoch_num, num_cls_ins, annealing_step, c_fisher, use_kl_div, device) + 
                        edl_red_loss(all_target[i], all_alpha[i], num_cls_ins, red_type, device))
                    )
                else:
                    ins_loss += torch.mean(
                        ins_weight * mse_loss(all_target[i], all_alpha[i], epoch_num, num_cls_ins, annealing_step, c_fisher, use_kl_div, device)
                    )

            loss = ins_loss / n_sample

        # pos: it is aggregated bag-level alpha
        # neg: it is instance-level alpha
        elif separate == 'II': 
            pos_alpha, pos_target, neg_alpha, neg_target = compute_auxiliary_output(output_alpha, target, separate=separate, 
                aggregate=aggregate, weight=weight)
            n_sample, pos_loss, neg_loss = 0, 0, 0
            
            # Weakly-Supervised signal for pos. bags
            if pos_alpha is not None:
                n_sample += len(pos_alpha)
                if red_type is not None and len(red_type) > 0:
                    pos_loss += torch.sum(
                        mse_loss(pos_target, pos_alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device) + 
                        edl_red_loss(pos_target, pos_alpha, num_classes, red_type, device) 
                    )
                else:
                    pos_loss += torch.sum(
                        mse_loss(pos_target, pos_alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device)
                    )
            
            # Supervised signal directly for neg. bags and instances
            if neg_alpha is not None:
                n_sample += len(neg_alpha)
                for i in range(len(neg_alpha)):
                    if red_type is not None and len(red_type) > 0:
                        neg_loss += torch.mean(
                            mse_loss(neg_target[i], neg_alpha[i], epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device) + 
                            edl_red_loss(neg_target[i], neg_alpha[i], num_classes, red_type, device) 
                        )
                    else:
                        neg_loss += torch.mean(
                            mse_loss(neg_target[i], neg_alpha[i], epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device)
                        )

            loss = (pos_loss + neg_loss) / n_sample

        # pos & neg: it is mean-aggregated bag-level alpha.
        elif separate == 'I': 
            # Weakly-Supervised signal for pos. & neg. bags
            alpha, target = compute_auxiliary_output(output_alpha, target, separate=separate, 
                aggregate=aggregate, weight=weight)
            if red_type is not None and len(red_type) > 0:
                loss = torch.mean(
                    mse_loss(target, alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device) + 
                    edl_red_loss(target, alpha, num_classes, red_type, device) 
                )
            else:
                loss = torch.mean(
                    mse_loss(target, alpha, epoch_num, num_classes, annealing_step, c_fisher, use_kl_div, device)
                )
        
        else:
            pass
    else:
        pass
    
    return loss

def get_edl_loss(loss_type, verbose=True):
    if loss_type == 'mse':
        if verbose:
            print("[info] loaded edl_mse_loss.")
        return edl_mse_loss
    else:
        return None

# Instance-level loss for EDL models
def edl_neg_instance_loss(ins_alpha, target, red_type='log-alpha', device=None):
    """
    Input:
        ins_alpha: the instance-level output of mini-batch bags, should be a list of Tensor [1, N, num_cls].
        target: the label of mini-batch bags, should be a Tensor with shape of [B, 1].
    """
    assert isinstance(ins_alpha, list)
    num_classes = ins_alpha[0].shape[-1]
    # ONLY consider negative bags
    neg_idx = torch.nonzero(target.squeeze() == 0).squeeze(-1).tolist()
    num_neg = len(neg_idx)
    if num_neg == 0:
        return 0
    
    neg_alpha = []
    select_instance = 'none'
    for i in neg_idx:
        cur_ins_alpha = ins_alpha[i].squeeze(0) # [N, num_cls]
        if select_instance == 'one_minconf':
            unc_idx = torch.argmin(cur_ins_alpha.detach().sum(-1, keepdims=True), 0)
            neg_alpha.append(cur_ins_alpha[unc_idx])
        else:
            neg_alpha.append(cur_ins_alpha)

    neg_target = [torch.zeros(x.shape[0], dtype=target.dtype, device=target.device).unsqueeze(-1) for x in neg_alpha] # list of Tensor [N, 1]

    neg_loss = 0
    for i in range(num_neg):
        neg_loss += torch.mean(edl_red_loss(neg_target[i], neg_alpha[i], num_classes, red_type, device))
    neg_loss = neg_loss / num_neg
    return neg_loss
