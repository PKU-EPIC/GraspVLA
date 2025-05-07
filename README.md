# GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data
[![arXiv](https://img.shields.io/badge/arXiv-2505.03233-df2a2a.svg)](https://arxiv.org/pdf/2505.03233)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://pku-epic.github.io/GraspVLA-web/)

<!-- [Shengliang Deng](https://shengliangd.github.io/about/), [Mi Yan](https://miyandoris.github.io/), [Songlin Wei](https://songlin.github.io/), Haixin Ma, Yuxin Yang, [Jiayi Chen](https://jychen18.github.io/), Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, [Heming Cui](https://i.cs.hku.hk/~heming/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/) -->

We present a cost-effective pretraining paradigm for training VLA models exclusively on large-scale synthetic data, achieving direct sim-to-real transfer and strong zero-shot generalizability for robotic grasping. Key contributions include:

- **SynGrasp-1B**: a billion-frame synthetic grasping dataset, spanning 240 object categories and 10,000+ objects.

- **GraspVLA**: a VLA model pretrained on SynGrasp-1B that achieves zero-shot generalization to real-world grasping without fine-tuning.

- **Unified CoT Framework**: GraspVLA integrates autoregressive perception and flow-matching-based action generation into a single reasoning process, enabling joint training on synthetic action data and internet-scale semantic data for open-vocabulary grasping.

![teaser](./figs/teaser.jpg)

TODO:
- [ ] Release the supplementary material
- [ ] Release the model weights
- [ ] Release the dataset

[![License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](LICENSE)