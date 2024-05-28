# MIREL: Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation

*International Conference on Machine Learning (ICML), 2024*

[[Paper]](https://arxiv.org/abs/2405.04405) | [[Project Page]](https://github.com/liupei101/MIREL) | [[Uncertainty Analysis]](https://github.com/liupei101/MIREL?tab=readme-ov-file#bag-level-and-instance-level-uncertainty-analysis)

**Note**: More materials, e.g., *project page* and *walkthrough notebooks*, are in preparation and will be released soon. Stay tuned.

**Abstract**: Uncertainty estimation (UE), as an effective means of quantifying predictive uncertainty, is crucial for safe and reliable decision-making, especially in high-risk scenarios. Existing UE schemes usually assume that there are completely-labeled samples to support fully-supervised learning. In practice, however, many UE tasks often have no sufficiently-labeled data to use, such as the Multiple Instance Learning (MIL) with only weak instance annotations. To bridge this gap, this paper, for the first time, addresses the weakly-supervised issue of Multi-Instance UE (MIUE) and proposes a new baseline scheme, *Multi-Instance Residual Evidential Learning* (MIREL). Particularly, at the fine-grained instance UE with only weak supervision, we derive a multi-instance residual operator through the Fundamental Theorem of Symmetric Functions. On this operator derivation, we further propose MIREL to jointly model the high-order predictive distribution at bag and instance levels for MIUE. Extensive experiments empirically demonstrate that our MIREL not only could often make existing MIL networks perform better in MIUE, but also could surpass representative UE methods by large margins, especially in instance-level UE tasks.

---

📚 Recent updates:
- 24/05/28: Upload experimental [files](https://github.com/liupei101/MIREL/blob/main/result/mirel-experiment/), [Notebook - Bag-level-Uncertainty-Analysis](https://github.com/liupei101/MIREL/blob/main/notebook/Bag-level_Uncertainty_Analysis.ipynb), and [Notebook - Instance-level-Uncertainty-Analysis](https://github.com/liupei101/MIREL/blob/main/notebook/Instance-level_Uncertainty_Analysis.ipynb).
- 24/05/27: Upload MIREL source codes

## Running the code

Using the following command to load configurations from a yaml file and train the model:
```bash
python3 main.py --config config/mnist-bags/cfg_abmil_mirel.yml --handler clf --multi_run
```

The configuration files for running MIREL models with different bag datasets (`MNIST-Bags`, `CIFAR10-Bags`, and `CAMELYON16`) are provided in [here](https://github.com/liupei101/MIREL/blob/main/config/). Detailed description of each configuration is commented in these files. 

All experimental files could be found in [here](https://github.com/liupei101/MIREL/blob/main/result/mirel-experiment/).

## Bag-level and instance-level uncertainty analysis

Please check
- Bag-level uncertainty analysis: [Notebook - Bag-level-Uncertainty-Analysis](https://github.com/liupei101/MIREL/blob/main/notebook/Bag-level_Uncertainty_Analysis.ipynb)
- Instance-level uncertainty analysis: [Notebook - Instance-level-Uncertainty-Analysis](https://github.com/liupei101/MIREL/blob/main/notebook/Instance-level_Uncertainty_Analysis.ipynb)

## Citation

If you find this work helps you more or less, please cite it via
```
@InProceedings{liu2024mirel,
  title={Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation},
  author={Liu, Pei and Ji, Luping},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
}
```
or `P. Liu and L. Ji, "Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation," in Proceedings of the 41st International Conference on Machine Learning, 2024.`.

Any issues can be sent via E-mail (yuukilp@163.com) or posted on the issue page.