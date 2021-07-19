# GNDAN
Codes of **GNDAN: Garph Navigated Dual Attention Network for Zero-Shot Learning** submitted to TNNLS. Note that this repository includes the trained model and test scripts, which is used for testing and checking our results reported in our paper. Once our paper is accepted, we will release all codes of this work.

## Preparing Dataset and Model
We provide trained models of three different datasets as follow, you can download them and store the model weight file and corresponding dataset into the `./saved_model` folder and `./data` folder respectively.
* CUB: [Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Model](https://drive.google.com/file/d/1RmA_mx-V7LaaqJ2cjbEefPOX9pSwGG8x/view?usp=sharing)
* SUN: [Dataset](http://cs.brown.edu/~gmpatter/sunattributes.html), [Model](https://drive.google.com/file/d/16n3_9T-l7ks5KYxMAyBQgW5msCZbaT44/view?usp=sharing)
* AWA2: [Dataset](http://cvml.ist.ac.at/AwA2/), [Model](https://drive.google.com/file/d/1y0hxUl5cRIoJJFXu3efb0RSwSVV0yw54/view?usp=sharing)

## Runing
Environment

```
python test.py --config config/test_CUB.json
python test.py --config config/test_SUN.json
python test.py --config config/test_AWA2.json
```

## Results
Results of our released model using various evaluation protocols on three datasets, both in conventional ZSL (CZSL) and generalized ZSL (GZSL) setting.

| Dataset | U | S | H | Acc |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 68.5 | 70.7 | 69.6 | 75.6 |
| SUN | 50.3 | 35.0 | 41.3 | 65.6 |
| AWA2 | 61.7 | 79.1 | 69.3 | 71.3 |

**Note**: All of above results run on a server with an AMD Ryzen 7 5800X CPU and a Nvidia RTX A6000 GPU.

## References
Parts of our codes based on [Attentive Region Embedding Network for Zero-shot Learning](https://github.com/gsx0/Attentive-Region-Embedding-Network-for-Zero-shot-Learning) and [Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention](https://github.com/hbdat/cvpr20_DAZLE).