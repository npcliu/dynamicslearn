# 2025.04.05更新：
之前的代码有问题，预测的是上次的力矩和加速度，所以他们的损失很小，版本回退到3月30号的，预测的是潜在空间的加速度。
# 2025.04.05更新：
添加了二阶段训练的代码，先训练正逆动力学（qddot<-->tau），再训练正运动学qddot->dq->q。不过这次提交没用到，还是用的全都
# 2025.04.04更新：
正逆动力学（qddot<-->tau）训练的很好，但是运动学qddot->dq->q训练的不好，误差有点大，目前是单步训练。我想固定动力学网络，进行二阶段训练，专门训运动学网络
# 2025.04.01更新
1)最好的mujoco逆动力学，力矩估计的特别准，也特别稳,最好的mujoco bachmark存到了"simplelog/sim2realmujoco20250331_234321"，因为没有在逆动力学中加入时间步长，但正动力学需要加，现在有个问题就是正动力学还是做不到像isaac那样小的误差，并且，我没有在q和v之间加入神经网络导致q的误差很大，v的误差由于加入了未建模神经网络导致误差比之前小得多了，但还是比isaac大得多

2) 写了个扩展卡尔曼滤波的程序，那个py文件里面有通过dq一阶差分求ddq的代码，同时筛选掉时间步长小于1毫秒的数据，然后q dq ddq都送到扩展卡尔曼中进行滤波。现在q和dq的过程噪声和观测噪声都很小，所以默认不对他们滤波，只对差分出的ddq进行滤波，不过确实一阶差分得到的ddq变化率好大
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
