import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_utils import *


#####################################################################################
#   MIREL (Multi-Instance Residual Evidential Learning) with classicial MIL pooling
#   * classicial MIL pooling contains Max- or Mean-based pooling
#####################################################################################
class DeepMIL(nn.Module):
    """
    Deep (Max- or Mean-based) Multiple Instance Learning for Bag-level Task.

    Args:
        dim_in: input instance dimension.
        dim_hid: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max'.
    """
    def __init__(self, dim_in=1024, dim_hid=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5, pooling='max', 
            pred_head='default', edl_output=False, ins_pred_from='bag', eins_frozen_feat=True, **kwargs):
        super(DeepMIL, self).__init__()
        assert pooling in ['mean', 'max']
        assert pred_head == 'default'
        assert ins_pred_from in ['bag', 'eins'] # use which branch for instance prediction
        self.ins_pred_from = ins_pred_from
        self.use_frozen_feat_in_eins = eins_frozen_feat
        self.edl_output = edl_output
        self.pooling = pooling

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
        
        if pred_head == 'default':
            self.pred_head = nn.Linear(dim_in, num_cls)
        else:
            self.pred_head = nn.Sequential(
                nn.Linear(dim_in, dim_hid),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(dim_hid, num_cls)
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
            print("[info] Initialized a DeepMIL model with EDL_head, and the pooling is {}.".format(self.pooling))
        else:
            print("[info] Initialized a DeepMIL model, and the pooling is {}.".format(self.pooling))

    def forward(self, X, ret_ins_res=False):
        """
        X: initial bag features, with shape of [b, K, d]
           where b = 1 for batch size, K is the instance size of this bag, and d is feature dimension.
        """
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        
        # global pooling
        # [B, N, C] -> [B, C]
        if self.pooling == 'mean':
            bag_emb = torch.mean(X, dim=1)
        elif self.pooling == 'max':
            bag_emb, _ = torch.max(X, dim=1)
        else:
            pass

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
            if self.ins_pred_from == 'bag': # directly predict instances using the bag clf head
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

######################################################################
#   MIREL (Multi-Instance Residual Evidential Learning) with DSMIL
######################################################################

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
    
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.output_class = output_class
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)

    def forward_with_single_instance(self, feats):
        # feats with shape of [N, K]
        feats = feats.unsqueeze(1) # [N, 1, K]
        V = self.v(feats) # [N, 1, V]
        A = torch.ones((V.shape[0], self.output_class, 1)).cuda() # [N, C, 1]
        B = torch.bmm(A, V) # [N, C, V]
        C = self.fcc(B).squeeze(-1) # [N, C, 1] -> [N, C]
        return C # [N, C]
        
    def forward(self, feats, c, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_hid=1024, num_cls=2, use_feat_proj=True, drop_rate=0.5, pooling='ds', 
            pred_head='default', edl_output=False, ins_pred_from='bag', eins_frozen_feat=True, **kwargs):
        super(DSMIL, self).__init__()
        assert pooling == 'ds'
        assert pred_head == 'default'
        assert ins_pred_from in ['ins', 'bag', 'eins'] # use which branch for instance prediction
        self.ins_pred_from = ins_pred_from
        self.use_frozen_feat_in_eins = eins_frozen_feat
        self.edl_output = edl_output
        self.pooling = pooling

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

        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_hid, num_cls)

        # Add modules for EDL and Instance enhancement
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
            print("[info] Initialized a DSMIL model with EDL_head.")
        else:
            print("[info] Initialized a DSMIL model.")

    def forward_with_single_instance(self, ins_feats):
        # ins_feats with shape of [N, K]
        out_bag_stream = self.b_classifier.forward_with_single_instance(ins_feats) # [N, C]
        _, out_ins_stream = self.i_classifier(ins_feats) # [N, C]
        out_ins_pred = 0.5 * (out_bag_stream + out_ins_stream) # [N, C]
        return out_ins_pred
        
    def forward(self, X, ret_ins_res=False):        
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, K]
            X = self.feat_proj(X)
        X = X.squeeze(0) # to [N, K] for input to i and b classifier

        feats, classes = self.i_classifier(X) # [N, K] for raw features, [N, C] for instance logits
        prediction_bag, A, B = self.b_classifier(feats, classes) # [1, C] for aggregated output
        max_prediction, _ = torch.max(classes, 0) # [1, C], the max logits for instance-level output
        bag_out = 0.5 * (prediction_bag + max_prediction) # logits = [1, C]

        if self.edl_output:
            bag_evidence = self.evidence_func(bag_out) # [1, 2]
            out = bag_evidence + 1 # the alpha parameters of dirichlet distribution
        else:
            out = bag_out

        if self.ins_enhance:
            X_feat = X.detach().clone() # [N, K]
            init_ins_out = self.forward_with_single_instance(X_feat) # [N, C]
            init_ins_evidence = self.evidence_func(init_ins_out).detach().clone() # frozen initital instance evidence
            if self.use_frozen_feat_in_eins:
                scale = self.ins_enhance_layer(X_feat) # [N, C] \in [-1, 1], frozen the instance features
            else:
                scale = self.ins_enhance_layer(X) # [N, C] \in [-1, 1]
            # enhanced_ins_evidence = (1 + scale) * init_ins_evidence # [N, C]
            enhanced_ins_evidence = init_ins_evidence ** (1 + scale) # [N, C]
            enhanced_ins_pred = enhanced_ins_evidence + 1
        else:
            enhanced_ins_pred = None

        if ret_ins_res:
            if self.ins_pred_from == 'ins':
                sel_cls = torch.argmax(out).item()
                ins_pred = A[:, sel_cls]
                ins_pred = (ins_pred - ins_pred.min()) / (ins_pred.max() - ins_pred.min())
                ins_pred = ins_pred.unsqueeze(0) # [1, N]
            elif self.ins_pred_from == 'bag': # directly predict instances using the bag clf head
                ins_out = self.forward_with_single_instance(X) # [N, C]
                if self.edl_output:
                    ins_pred = self.evidence_func(ins_out) + 1 # [N, C]
                else:
                    ins_pred = F.softmax(ins_out, dim=-1)[:, 1] # [N, ]
                ins_pred = ins_pred.unsqueeze(0) # [1, N, C] or [1, N]
            elif self.ins_pred_from == 'eins':
                init_ins_evidence = init_ins_evidence.unsqueeze(0)
                enhanced_ins_pred = enhanced_ins_pred.unsqueeze(0)
                ins_pred = (init_ins_evidence + 1, enhanced_ins_pred) # tuple of [1, N, C]
            else:
                pass
            return out, ins_pred # [1, C], ins_prediction
        else:
            return out
