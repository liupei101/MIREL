import torch 
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import *


#####################################################################
#   MIREL (Multi-Instance Residual Evidential Learning) with ABMIL
#####################################################################
class ABMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task. (pooling = attention)

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, default by attention pooling.
    """
    def __init__(self, dim_in=1024, dim_hid=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5, pooling='attention', 
            pred_head='default', edl_output=False, ins_pred_from='ins', eins_frozen_feat=True, **kwargs):
        super(ABMIL, self).__init__()
        assert pooling in ['attention', 'gated_attention']
        assert pred_head == 'default'
        assert ins_pred_from in ['ins', 'bag', 'eins'] # use which branch for instance prediction
        self.ins_pred_from = ins_pred_from
        self.use_frozen_feat_in_eins = eins_frozen_feat
        self.edl_output = edl_output

        if use_feat_proj == 'default':
            self.feat_proj = Feat_Projecter(dim_in, dim_in, drop_rate=drop_rate)
        elif use_feat_proj == 'mnist':
            self.feat_proj = MNIST_Feat_Projecter(dim_in, dim_in, drop_rate=drop_rate)
        elif use_feat_proj == 'cifar10':
            self.feat_proj = CIFAR_Feat_Projecter(dim_in, dim_in, drop_rate=drop_rate)
        elif use_feat_proj == 'gau':
            self.feat_proj = Gau_Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        
        self.pooling = pooling
        if pooling == 'attention':
            self.ins_layer = Scoring_Net(dim_in, dim_hid)
        elif pooling == 'gated_attention':
            self.ins_layer = Gated_Scoring_Net(dim_in, dim_hid)
        else:
            self.ins_layer = None
        
        if pred_head == 'default':
            self.pred_head = nn.Linear(dim_in, 2)
        else:
            self.pred_head = nn.Sequential(
                nn.Linear(dim_in, dim_emb),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(dim_emb, 2)
            )
        
        edl_evidence_func = kwargs['edl_evidence_func']
        if edl_evidence_func == 'relu':
            self.evidence_func = relu_evidence
        elif edl_evidence_func == 'exp':
            self.evidence_func = exp_evidence
        elif edl_evidence_func == 'softplus':
            self.evidence_func = softplus_evidence
        elif edl_evidence_func == 'elu':
            self.evidence_func = elu_evidence
        else:
            self.evidence_func = None
            
        # build an instance enhancement layer
        self.ins_enhance = self.ins_pred_from == 'eins'
        if self.ins_enhance:
            assert self.edl_output, "Instance enhancement can be used only in EDL models."
            self.ins_enhance_layer = nn.Sequential(
                nn.Linear(dim_in, dim_hid),
                nn.Tanh(), 
                nn.Linear(dim_hid, 2),
                nn.Tanh(), 
            )
            print("[info] Initialized an instance enhancement layer: frozen_feat = {}.".format(self.use_frozen_feat_in_eins))
        else:
            self.ins_enhance_layer = None

        if self.edl_output:
            print("[info] Initialized an ABMIL model with EDL_head, and the pooling is {}.".format(self.pooling))
        else:
            print("[info] Initialized an ABMIL model, and the pooling is {}.".format(self.pooling))

    def forward(self, X, ret_ins_res=False):
        """
        X: initial bag features, with shape of [B, N, C]
           where B = 1 for batch size, N is the instance size of this bag, and C is feature dimension.
        """
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        
        # global pooling
        # [B, N, C] -> [B, C]
        if self.pooling == 'attention' or self.pooling == 'gated_attention':
            A = self.ins_layer(X) # [B, 1, N]
        else:
            A = None
        
        bag_emb = torch.matmul(A, X) # [B, 1, N] bmm [B, N, C] = [B, 1, C]
        bag_emb = bag_emb.squeeze(1) # [B, 1, C] -> [B, C]

        # bag clf head: [B, C] -> [B, 2]
        if self.edl_output:
            bag_out = self.pred_head(bag_emb)
            bag_evidence = self.evidence_func(bag_out) # [B, 2]
            out = bag_evidence + 1 # the alpha parameters of dirichlet distribution
        else:
            out = self.pred_head(bag_emb) # [B, 2]

        if self.ins_enhance:
            X_feat = X.detach().clone()
            init_ins_evidence = self.evidence_func(self.pred_head(X_feat)).detach().clone() # frozen initital instance evidence
            if self.use_frozen_feat_in_eins:
                scale = self.ins_enhance_layer(X_feat) # [B, N, 2] \in [-1, 1], frozen the instance features
            else:
                scale = self.ins_enhance_layer(X) # [B, N, 2] \in [-1, 1]
            # enhanced_ins_evidence = (1 + scale) * init_ins_evidence # [B, N, 2]
            enhanced_ins_evidence = init_ins_evidence ** (1 + scale) # [B, N, 2]
            enhanced_ins_pred = enhanced_ins_evidence + 1
        else:
            enhanced_ins_pred = None

        if ret_ins_res:
            if self.ins_pred_from == 'ins':
                if self.pooling == 'attention':
                    ins_pred = A.squeeze(1) # [B, N]
                    ins_pred = (ins_pred - ins_pred.min()) / (ins_pred.max() - ins_pred.min())
                else:
                    ins_pred = ins_prob.squeeze(1) # [B, N]
            elif self.ins_pred_from == 'bag': # directly predict instances using the bag clf head
                if self.edl_output:
                    ins_pred = self.evidence_func(self.pred_head(X)) + 1 # [B, N, 2]
                else:
                    ins_pred = F.softmax(self.pred_head(X), dim=-1)[:, :, 1] # [B, N]
            elif self.ins_pred_from == 'eins':
                ins_pred = (init_ins_evidence + 1, enhanced_ins_pred) # tuple of [B, N, 2]
            else:
                pass
            return out, ins_pred # [B, 2], ins_prediction
        else:
            return out
