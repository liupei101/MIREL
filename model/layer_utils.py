import torch
import torch.nn as nn
import torch.nn.functional as F


################################################
#    Common Evidential Activation Functions 
################################################
def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def elu_evidence(y):
    return F.elu(y) + 1 # evidence >= 0


###################################################
#   Feature Projection Class for Patch Embedding
###################################################
class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, drop_rate=0):
        super(Feat_Projecter, self).__init__()
        if drop_rate > 1e-5:
            self.projecter = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.Dropout(drop_rate),
                nn.ReLU(),
            )
        else:
            self.projecter = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
            )

    def forward(self, x):
        # x = [B, N, C]
        assert len(x.shape) == 3
        B, N = x.shape[0], x.shape[1]
        x = x.flatten(0, 1) # forced the dimension be [B*N, C]
        feat = self.projecter(x) # [B*N, C]
        ret_feat = feat.reshape(B, N, -1) # [B, N, C]
        return ret_feat


class Gau_Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Gau_Feat_Projecter, self).__init__()
        dim_input = 2 # inputs are 2D points for vis
        # project 2D points to high dimension space
        self.projecter = nn.Sequential(
            nn.Linear(dim_input, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x = [B, N, C] or [N, C]
        ret_feat = self.projecter(x)
        return ret_feat

"""
LeNet for MNIST feature extractor.
Codes roughly follows the network implementation used in ABMIL.
"""
class MNIST_Feat_Projecter(nn.Module):
    def __init__(self, in_dim=800, out_dim=500, drop_rate=0):
        super(MNIST_Feat_Projecter, self).__init__()
        if drop_rate <= 1e-5:
            drop_rate = 0
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.Dropout2d(drop_rate),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(drop_rate),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        assert in_dim == 50 * 4 * 4 # out size is 4 * 4
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape = [B, N, 1, 28, 28]
        B, N = x.shape[0], x.shape[1]
        x = x.flatten(0, 1) # forced the dimension be [B*N, 1, 28, 28]
        feat1 = self.feature_extractor_part1(x) # [B*N, 50, 4, 4]
        feat1 = feat1.view(feat1.shape[0], -1) # [B*N, 50 * 4 * 4]
        feat2 = self.projecter(feat1) # [B*N, 800]
        feat2 = feat2.reshape(B, N, -1) # [B, N, 800]
        return feat2

"""
AlexNet for CIFAR-10 feature extractor.

Codes are adapted from https://github.com/facebookresearch/deepcluster/blob/main/models/alexnet.py
"""
def make_layers_features(config, input_dim=3, bn=False, drop_rate=0):
    # (number of filters, kernel size, stride, pad)
    CFG = {
        'big': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'],
        'small': [(64, 11, 4, 2), 'M', (192, 5, 1, 2), 'M', (384, 3, 1, 1), (256, 3, 1, 1), (256, 3, 1, 1), 'M'],
        'mnist': [(32, 6, 2, 2), (64, 3, 1, 1), 'M', (128, 3, 1, 1), (128, 3, 1, 1), 'M'],
        'CAMELYON': [(96, 12, 4, 4), (256, 12, 4, 4), 'M_', (256, 5, 1, 2), 'M_', (512, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1), 'M_'],
        'CIFAR10': [(96, 3, 1, 1), 'M', (192, 3, 1, 1), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (192, 3, 1, 1), 'M'],
    }
    CFG_Dropout = {
        'CIFAR10': [False, False, False, True, True],
    }
    cfg = CFG[config]
    cfg_dropout = CFG_Dropout[config]
    layers = []
    in_channels = input_dim
    idx_conv_block = 0
    for v in cfg:
        if v == '_M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        elif v == 'M_':
            layers += [nn.MaxPool2d(kernel_size=4, stride=2, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            layers += [conv2d]
            if bn:
                layers += [nn.BatchNorm2d(v[0])]
            if drop_rate > 1e-5:
                if cfg_dropout[idx_conv_block]:
                    layers += [nn.Dropout2d(drop_rate)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = v[0]
            idx_conv_block += 1
    return nn.Sequential(*layers)


class CIFAR_Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, drop_rate=0):
        super(CIFAR_Feat_Projecter, self).__init__()
        self.feature_extractor_part1 = make_layers_features('CIFAR10', input_dim=3, bn=True, drop_rate=drop_rate)
        in_dim = 192 * 3 * 3
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape = [B, N, 3, 32, 32]
        B, N = x.shape[0], x.shape[1]
        x = x.flatten(0, 1) # forced the dimension be [B*N, 3, 32, 32]
        feat1 = self.feature_extractor_part1(x)
        feat1 = feat1.view(feat1.shape[0], -1)
        feat2 = self.projecter(feat1) # [B*N, out_dim]
        feat2 = feat2.reshape(B, N, -1) # [B, N, out_dim]
        return feat2


###################################################
#    Some Useful Attention layer used in ABMIL
###################################################
class Gated_Scoring_Net(nn.Module):
    """
    Refer to [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim, hid_dim, dropout=0):
        super(Gated_Scoring_Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        # x -> out : [B, N, d] -> [B, d]
        emb = self.fc1(x) # [B, N, d']
        scr = self.score(x) # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        A_ = self.fc2(new_emb) # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1) # [B, 1, N]
        score = F.softmax(A_, dim=2) # [B, 1, N]
        return score


class Scoring_Net(nn.Module):
    """
    Refer to [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim=1024, hid_dim=512, out_dim=1):
        super(Scoring_Net, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        # x -> out : [B, N, d] -> [B, d]
        A_ = self.attention(x)  # [B, N, O]
        A_ = torch.transpose(A_, 2, 1)  # [B, O, N]
        score = F.softmax(A_, dim=2)  # [B, O, N]
        return score


class Gated_Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim, hid_dim, dropout=0.5):
        super(Gated_Attention_Pooling, self).__init__()
        self.scoring_layer = Gated_Scoring_Net(in_dim, hid_dim, dropout)

    def forward(self, x, ret_attn=True):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        A = self.scoring_layer(x) # [B, 1, N]
        out = torch.matmul(A, x).squeeze(1) # [B, 1, d]
        if ret_attn:
            A = A.squeeze(1)
            return out, A
        return out


class Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by 
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim=1024, hid_dim=512):
        super(Attention_Pooling, self).__init__()
        self.scoring_layer = Scoring_Net(in_dim, hid_dim)

    def forward(self, x, ret_attn=True):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        A = self.scoring_layer(x) # [B, 1, N]
        out = torch.matmul(A, x).squeeze(1)  # [B, 1, N] bmm [B, N, d] = [B, 1, d]
        if ret_attn:
            A = A.squeeze(1)
            return out, A
        return out
