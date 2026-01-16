# PQGNet: Perceptual Query Guided Network for Infrared Small Target Detection (TGRS 2026)

🎉 **Important Notice: This paper has been officially accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS) on January 12, 2026!**


## 📄 Paper Information

<div align="center">
  <img src="./asset/PQGNet.drawio.png">
</div>
<p align="center">
  The network structure diagram of PQGNet. Our PQGNet adopts a U-shaped structure and constructs a Perceptual Query Supervision Mechanism(PQSM).
</p>

### Abstract:
**Infrared small target detection (IRSTD) holds critical importance for military security applications. Although U-shaped architectures have improved baseline performance, existing methods still suffer from two key limitations: 1) Insufficient spatial perception for tiny targets leads to target location loss.; 2) Edge degradation and semantic ambiguity in deep feature reconstruction.
To address these challenges, we propose PQGNet with the following contributions: 
To enhance capability of spatial perception and improve feature fusion guidance, we introduce the Perceptual Query Supervision Mechanism (PQSM), which utilizes perceptual loss to constrain spatial feature learning of each encoder layer. The Perceptual Feature Construction Module (PFCM) constructs enhanced perceptual features to preserve target localization information, while the Perceptual Query Guidance Module (PQGM) adopts cross-attention to guide global and regional feature queries through skip connections, optimizing target feature extraction. To mitigate reconstruction degradation and semantic ambiguity, distinct from existing wavelet-based approaches that simply substitute pooling layers, we design a Max pooling-Wavelet Hybrid Layer (MWHL) and High-frequency Enhancement Wavelet Layer (HEWL) that exploit discrete wavelet transform properties to enhance deep semantic representations using shallow high-frequency details. Comprehensive experiments on NUDT-SIRST, NUAA-SIRST, and IRSTD-1K datasets demonstrate that PQGNet significantly surpasses state-of-the-art methods in detection performance, while maintaining a competitive balance between computational complexity and accuracy.Our code will be made public at https://github.com/PepperCS/PQGNet.**




## 🚀 Code Installation

We use [Pytorch2.6.0 + CUDA11.8] as  a software configuration environment.

### Environment Installation
```
conda create -n PQGNet python=3.10.18
conda activate PQGNet
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


### Datasets
- IRSTD-1K:[[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
- NUDT-SIRST:[[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
- NUAA-SIRST:&nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  ```

## 📊 Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{11352975,
  author={Liu, Pingping and Li, Aohua and Lu, Yubing and Zhang, Tongshun and Yang, Ming and Zhou, Qiuzhan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={PQGNet: Perceptual Query Guided Network for Infrared Small Target Detection}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Image edge detection;Object detection;Semantics;Transformers;Location awareness;Degradation;Discrete wavelet transforms;Standards;Geoscience and remote sensing;Infrared Small Target Detection (IRSTD);Convolutional Neural Network (CNN);Cross Attention;Wavelet Transform;U-Net Architecture;Perceptual Loss},
  doi={10.1109/TGRS.2026.3654433}}

```

## 



Please ⭐ **Star** this repository to receive the latest update notifications!
