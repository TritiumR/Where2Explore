# Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects [NeuriIPS 2023]


![Overview](/images/guide.png)

**Where2Explore framework: Given an articulated 3D object, Although Affordance fails to directly generalize to 
novel categories (Left) via only a few interactions on low-similarity areas (Middle), our framework could learn the semantic information 
on novel objects (Right).**

## Introduction
Articulated object manipulation is a fundamental yet challenging task in robotics. Due to significant geometric and semantic 
variations across object categories, previous manipulation models struggle to generalize to novel categories. Few-shot 
learning is a promising solution for alleviating this issue by allowing robots to perform a few interactions with unseen 
objects. However, extant approaches often necessitate costly and inefficient test-time interactions with each unseen instance. 
Recognizing this limitation, we observe that despite their distinct shapes, different categories often share similar local 
geometries essential for manipulation - a factor typically underutilized in 
previous few-shot learning works. To harness this commonality, we introduce ‘Where2Explore’, an affordance learning framework 
that effectively explores novel categories with minimal interactions on a limited number of instances. Our framework 
explicitly estimates the geometric similarity across different categories, identifying local areas that differ from shapes 
in the training categories for efficient exploration while concurrently transferring affordance knowledge to similar parts 
of the objects. 
## About the paper

Our team: 
[Chuanruo Ning](https://tritiumr.github.io),
[Ruihai Wu](https://warshallrho.github.io),
Haoran Lu,
[Kaichun Mo](https://kaichun-mo.github.io),
and [Hao Dong](https://zsdonghao.github.io)
from 
Peking University and NVIDIA.

Arxiv Version: https://arxiv.org/abs/2309.07473

## About this repository

This repository provides data and code as follows.


```
    data/                   # contains data, models, results, logs
    code/                   # contains code and scripts
         # please follow `code/README.md` to run the code
    stats/                  # contains helper statistics
```

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## License

MIT Licence
