# MIREL

[**ICML 2024**] This is the official implementation of MIREL: Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation.

The paper, code, and project page are in preparation and will be released soon. 

Stay tuned.

[[OpenReview]](https://openreview.net/forum?id=cxiqxDnrCx) | [[arXiv]](https://arxiv.org) | [[Project Page]](https://github.com/liupei101/MIREL)
## Overview

*TL;DR*: This paper addresses the issue of multi-instance uncertainty estimation for the first time and proposes a new baseline scheme, Multi-Instance Residual Evidential Learning, to jointly quantify predictive uncertainty at bag and instance levels in MIL.

*Abstract*:
> Uncertainty estimation (UE), as an effective means of quantifying predictive uncertainty, is crucial for safe and reliable decision-making, especially in high-risk scenarios. Existing UE schemes usually assume that there are completely-labeled samples to support fully-supervised learning. In practice, however, many UE tasks often have no sufficiently-labeled data to use, such as the Multiple Instance Learning (MIL) with only weak instance annotations. To bridge this gap, this paper, for the first time, addresses the weakly-supervised issue of Multi-Instance UE (MIUE) and proposes a new baseline scheme, *Multi-Instance Residual Evidential Learning* (MIREL). Particularly, at the fine-grained instance UE with only weak supervision, we derive a multi-instance residual operator through the Fundamental Theorem of Symmetric Functions. On this operator derivation, we further propose MIREL to jointly model the high-order predictive distribution at bag and instance levels for MIUE. Extensive experiments empirically demonstrate that our MIREL not only could often make existing MIL networks perform better in MIUE, but also could surpass representative UE methods by large margins, especially in instance-level UE tasks.


## Citation

Any issues can be sent via E-mail (yuukilp@163.com) or posted on the issue page of this repo.

If you find this work helps you more or less, please cite it via
```
@InProceedings{liu2024mirel,
  title={Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation},
  author={Liu, Pei and Ji, Luping},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2024},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
}
```
or `P. Liu and L. Ji, “Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation,” in Proceedings of the 40th International Conference on Machine Learning, 2024.`.
