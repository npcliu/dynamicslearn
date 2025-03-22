# 2025.03.22更新
使用isaacgym采集的固定基数据，状态量为10个关节角，仅有正动力学，先保存一下，以便后续更改。同时保存一下训练结果（simplelog/seed=21,pred_horizon=10,batch_size=1000,lr=0.0001,ntrajs=8192,uscale=10.0,dt=0.001）

# MechaModLearn

Authors: [@kunalmenda](https://github.com/kunalmenda) and [@rejuvyesh](https://github.com/rejuvyesh)

This library provides a structured framework for learning mechanical systems in PyTorch.

---

## Installation

Requires Python3.

```
git clone https://github.com/sisl/mechamodlearn.git
cd mechamodlearn
pip install -e .
```

## Usage
Example experiments are placed in [`experiments`](./experiments) directory.

To run the Simple experiment:

```
python experiments/simple.py --logdir /tmp/simplelog
```

## References

---
If you found this library useful in your research, please consider citing our [paper](https://arxiv.org/abs/1902.08705):
```
@article{gupta2019mod,
    title={A General Framework for Structured Learning of Mechanical Systems},
    author={Gupta, Jayesh K. and Menda, Kunal and Manchester, Zachary and Kochenderfer, Mykel},
    journal={arXiv preprint arXiv:1902.08705},
    year={2019}
}
```
