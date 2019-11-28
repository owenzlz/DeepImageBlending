# Deep Image Blending
This is a Pytorch implementation of our paper "Deep Image Blending". 

[Deep Image Blending](https://arxiv.org/pdf/1910.11495.pdf) <br />
[Lingzhi Zhang](https://owenzlz.github.io/), Tarmily Wen, [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)  <br />
GRASP Laboratory, University of Pennsylvania

 

## Introduction

We propose a Poisson blending loss that achieves the same purpose of Poisson Image Editing. We jointly optimize the proposed Poisson blending loss with style and content loss computed from a deep network, and reconstruct the blending region by iteratively updating the pixels using the L-BFGS solver. In the blending image, we not only smooth out gradient domain of the blending boundary but also add consistent texture into the blending region.

<img src='demo_imgs/first_demo.png' align="middle" width=540>

## Usage

[TODO]

## Ablation Study

<img src='demo_imgs/ablation_study.png' align="middle" width=720>


## Example results for paintings

<img src='demo_imgs/painting_comparison.png' align="middle" width=720>


## Example results for real-world images

<img src='demo_imgs/real_comparison.png' align="middle" width=720>


## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/1910.11495.pdf):

```
@article{zhang2019deep,
  title={Deep Image Blending},
  author={Zhang, Lingzhi and Wen, Tarmily and Shi, Jianbo},
  journal={arXiv preprint arXiv:1910.11495},
  year={2019}
}
```
