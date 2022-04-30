# GNDAN
<!-- ![](imgs/model.jpeg) -->
Codes for the paper "**GNDAN: Graph Navigated Dual Attention Network for Zero-Shot Learning**" accepted to TNNLS. Note that this repository includes the trained model and test scripts, which is used for testing and checking our results reported in our paper.

## Preparing Dataset and Model
We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1RjzIVQ9YykhOusAcjM9QHlMp5W_iaQoY?usp=sharing)) of three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/). You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── CUB_GNDAN_weights.pth
│   ├── SUN_GNDAN_weights.pth
│   └── AWA2_GNDAN_weights.pth
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```
## Requirements
The code implementation of **GNDAN** mainly based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). All of our experiments run and test in Python 3.8.8. To install all required dependencies:
```
$ pip install -r requirements.txt
```
## Runing
Runing following commands and testing **GNDAN** on different dataset: 
```
$ python test.py --config config/test_CUB.json      #CUB
$ python test.py --config config/test_SUN.json      #SUN
$ python test.py --config config/test_AWA2.json     #AWA2
```

## Results
Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings. These released results are slightly higher than the results in the paper.

| Dataset | U | S | H | Acc |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 68.5 | 70.7 | 69.6 | 75.6 |
| SUN | 50.3 | 35.0 | 41.3 | 65.6 |
| AWA2 | 61.7 | 79.1 | 69.3 | 71.3 |

**Note**: All of above results are run on a server with an AMD Ryzen 7 5800X CPU and one Nvidia RTX A6000 GPU.

## Citation
If this work is helpful for you, please cite our paper.

```
@article{chen2021gndan,
    author    = {Chen, Shiming and Hong, Ziming and Xie, Guo-Sen and Peng, Qinmu and You, Xinge and Ding, Weiping and Shao, Ling},
    title     = {GNDAN: Graph Navigated Dual Attention Network for Zero-Shot Learning},
    journal = {IEEE Transactions on Neural Networks and Learning Systems},
    year      = {2022}
}
```

## References
Parts of our codes based on:
* [gsx0/Attentive-Region-Embedding-Network-for-Zero-shot-Learning](https://github.com/gsx0/Attentive-Region-Embedding-Network-for-Zero-shot-Learning)
* [hbdat/cvpr20_DAZLE](https://github.com/hbdat/cvpr20_DAZLE)
