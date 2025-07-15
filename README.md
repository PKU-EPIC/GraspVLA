
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=it">Itapano</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=PKU-EPIC&project=GraspVLA&lang=id">Bahasa Indonesia</a>
      </div>
    </div>
  </details>
</div>

# GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data
[![arXiv](https://img.shields.io/badge/arXiv-2505.03233-df2a2a.svg)](https://arxiv.org/pdf/2505.03233)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://pku-epic.github.io/GraspVLA-web/)

<!-- [Shengliang Deng](https://shengliangd.github.io/about/), [Mi Yan](https://miyandoris.github.io/), [Songlin Wei](https://songlin.github.io/), Haixin Ma, Yuxin Yang, [Jiayi Chen](https://jychen18.github.io/), Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, [Heming Cui](https://i.cs.hku.hk/~heming/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/) -->

We present a cost-effective pretraining paradigm for VLA models using only synthetic data, achieving direct sim-to-real transfer and strong zero-shot generalizability for robotic grasping. Key contributions include:

- **SynGrasp-1B**: a billion-frame synthetic grasping dataset, spanning 240 object categories and 10,000+ objects.

- **GraspVLA**: a VLA model pretrained on SynGrasp-1B that achieves zero-shot generalization to real-world grasping without fine-tuning.

- **Unified CoT Framework**: GraspVLA integrates autoregressive perception and flow-matching-based action generation into a single reasoning process, enabling joint training on synthetic action data and internet-scale semantic data for open-vocabulary grasping.

![teaser](./figs/teaser.jpg)

TODO List:
- [ ] Release the supplementary material
- [ ] Release model weights
- [ ] Release SynGrasp-1B dataset

[![License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](LICENSE)