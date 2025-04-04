#!/usr/bin/env python3
#
# File: simple.py
#
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch

from mechamodlearn import dataset, utils, viz_utils, rigidbody, models
from mechamodlearn.trainer import OfflineTrainer
from mechamodlearn.systems import ActuatedDampedPendulum, DampedMultiLinkAcrobot, DEFAULT_SYS_PARAMS
from mechamodlearn.rigidbody import LearnedRigidBody
import h5py


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
print(DEVICE)
SYSTEMS = {
    'dampedpendulum': ActuatedDampedPendulum,
    '2linkdampedacrobot': lambda p: DampedMultiLinkAcrobot(2, p),
}

class ExternalData:
    def __init__(self, qdim, udim, device):

        self._qdim = qdim
        self._udim = udim
        self._device = device
        self.thetamask = torch.ones(10, device=DEVICE)
        print('self.thetamask', self.thetamask)


# def get_dataset(system, T: float, dt: float, ntrajs: int, uscale: float, qrange=(-1, 1),
#                 vrange=(-10, 10)):

#     t_points = torch.arange(0, T, dt).requires_grad_(True)
#     q0 = torch.stack([torch.empty(system._qdim).uniform_(*qrange)
#                       for _ in range(ntrajs)]).requires_grad_(True)
#     v0 = torch.stack([torch.empty(system._qdim).uniform_(*vrange)
#                       for _ in range(ntrajs)]).requires_grad_(True)
#     u_T_B = torch.randn(len(t_points), ntrajs, system._udim) * uscale

#     data = dataset.ActuatedTrajectoryDataset.FromSystem(system, q0, v0, u_T_B, t_points)
#     # print(data.q_B_T.shape)
#     return data

def get_dataset(system, T, dt, h5_file_path):
    t_points = torch.arange(0, T, dt, device=DEVICE).requires_grad_(True)
    if h5_file_path==None:
        train_data = None
        valid_data = None
        test_data = None
        print('please input h5 file!!!')
    else:
        with h5py.File(h5_file_path, 'r') as f:
            timestamps_ = f['timestamps'][:]
            q_ = f['q'][:].squeeze(axis=1)
            dq_ = f['dq'][:].squeeze(axis=1)
            ddq_ = f['ddq'][:].squeeze(axis=1)
            tau_ = f['tau'][:].squeeze(axis=1)
            
        timestamps = timestamps_[:-1]
        q = q_[:-1]
        dq = dq_[:-1]
        ddq = ddq_[1:] #后向差分得到的，所以第一个对应于第0个的加速度
        tau = tau_[:-1]
        # 时间差计算（向量化运算优化）
        time_diff = timestamps[1:] - timestamps[:-1]
        # print(time_diff[0:10])
        # print(min(time_diff))
        invalid_indices = np.where(time_diff < 0.0008)[0] + 1  # 标记异常点
        
        # 构建掩码数组
        mask = np.ones(len(timestamps), dtype=bool)
        mask[invalid_indices] = False

        # 数据过滤 过滤掉时间差值小于1毫秒的数据
        timestamps_cpu = timestamps[mask]
        # time_diff = timestamps_cpu[1:] - timestamps_cpu[:-1]
        # print(min(time_diff))
        q_cpu = q[mask]
        dq_cpu = dq[mask]
        ddq_cpu = ddq[mask]
        tau_cpu = tau[mask]
        
        # 重新计算时间差用于加速度计算
        time_diff_valid = timestamps_cpu[1:] - timestamps_cpu[:-1]
        dq_diff = dq_cpu[1:] - dq_cpu[:-1]
        ddq_cpu_computed = dq_diff / time_diff_valid
        
        # 为使 ddq 与其他数据维度一致，在第一个时刻填入第一个计算值（也可以设为 0）
        ddq_cpu_computed = np.vstack((ddq_cpu_computed, ddq_cpu_computed[-1, :]))
            
        timestamps_gpu = torch.from_numpy(timestamps_cpu).to(system._device) 
        q_gpu = torch.from_numpy(q_cpu).to(system._device) 
        dq_gpu = torch.from_numpy(dq_cpu).to(system._device) 
        ddq_gpu = torch.from_numpy(ddq_cpu).to(system._device) 
        ddq_gpu_computed = torch.from_numpy(ddq_cpu_computed).to(system._device) 
        tau_gpu = torch.from_numpy(tau_cpu).to(system._device) 
        
        print(tau_cpu[10:20, 0])
        print(tau_gpu[10:20, 0])
        
        len_t = len(t_points)
        N = q_gpu.shape[0]
        timestamps_gpu_t = torch.stack([timestamps_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        q_gpu_t = torch.stack([q_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        dq_gpu_t = torch.stack([dq_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        ddq_gpu_t = torch.stack([ddq_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        # ddq_gpu_t = torch.stack([ddq_gpu_computed[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        tau_gpu_t = torch.stack([tau_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0) 
        q_gpu_t = utils.wrap_to_pi(q_gpu_t.view(-1, system._qdim), system.thetamask).view(
            len_t, -1, system._qdim) 
        
        print(tau_gpu_t[10:20, 0, 0])
        # print(dq_gpu_t[0:10, 0, 0])
        # print(tau_gpu_t[0:10, 0, 0])
        
        n_samples = q_gpu_t.shape[1]  # 获取样本数量 89997
        device = q_gpu_t.device  # 保持设备一致性
        # 1. 生成随机索引
        indices = torch.randperm(n_samples, device=device)
        train_size = int(0.8 * n_samples)  # 80%训练
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        # 2. 按索引划分数据集（保持时间步维度）
        def split_data(tensor):
            return tensor[:, train_idx], tensor[:, test_idx]

        timestamps_train, timestamps_test = split_data(timestamps_gpu_t)
        q_train, q_test = split_data(q_gpu_t)
        dq_train, dq_test = split_data(dq_gpu_t)
        ddq_train, ddq_test = split_data(ddq_gpu_t)
        tau_train, tau_test = split_data(tau_gpu_t)

        train_data = (timestamps_train, q_train, dq_train, ddq_train, tau_train)
        valid_data = (timestamps_test, q_test, dq_test, ddq_test, tau_test)
        test_data = (timestamps_gpu_t[:, 650:750], q_gpu_t[:, 650:750, :], 
                     dq_gpu_t[:, 650:750, :], ddq_gpu_t[:, 650:750, :], tau_gpu_t[:, 650:750, :])
    train_dataset = dataset.ActuatedTrajectoryDataset.FromExternalData(train_data)
    valid_dataset = dataset.ActuatedTrajectoryDataset.FromExternalData(valid_data)
    test_dataset = dataset.ActuatedTrajectoryDataset.FromExternalData(test_data)
    # print(data.q_B_T.shape)
    return train_dataset, valid_dataset, test_dataset

def train(seed: int, dt: float, pred_horizon: int, num_epochs: int, batch_size: int,
          lr: float, ntrajs: int, uscale: float, logdir: str):
    args = locals()
    args.pop('logdir')
    args.pop('num_epochs')
    exp_name = ",".join(["=".join([key, str(val)]) for key, val in args.items()])

    utils.set_rng_seed(seed)

    # system = SYSTEMS[system](DEFAULT_SYS_PARAMS[system])

    # train_dataset = get_dataset(system, pred_horizon * dt, dt, ntrajs, uscale)
    # valid_dataset = get_dataset(system, pred_horizon * dt, dt, ntrajs, uscale)
    # test_dataset = get_dataset(system, 4, dt, 4, uscale)
    
    system = ExternalData(10, 10, DEVICE)
    # train_dataset, valid_dataset, test_dataset = get_dataset(system, pred_horizon * dt, dt, 
    #         'isaacdataset.h5') #isaacdataset
    train_dataset, valid_dataset, test_dataset = get_dataset(system, pred_horizon * dt, dt, 
            'sim2realmujocodataset.h5') 
    print(train_dataset.q_B_T[0])

    model = LearnedRigidBody(system._qdim, system._udim, system.thetamask)
    # potential = models.DelanZeroPotential(1)
    # model = rigidbody.DeLan(1, 32, 3, torch.tensor([1, 0]), udim=1, potential=potential)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=batch_size, gamma=0.5)
    
    if logdir is not None:
        logdir = Path(logdir) / '{:%Y%m%d_%H%M%S}'.format(datetime.now())

    trainer = OfflineTrainer(model, opt, dt, train_dataset, valid_dataset, learning_rate_scheduler=scheduler, 
                             pred_horizon=pred_horizon, num_epochs=num_epochs, batch_size=batch_size, vlambda=1, log_viz=False, 
                             ckpt_interval=100, summary_interval=200, shuffle=True, integration_method='euler', 
                             logdir=logdir, device=DEVICE)

    metrics = trainer.train()

    if logdir is not None:
        torch.save(model, Path(logdir) / 'metrics_{:%Y%m%d-%H%M%S}.pt'.format(datetime.now()))
    return metrics


@click.command()
@click.option('--seed', default=21, type=int)
@click.option('--dt', default=0.001, type=float)
@click.option('--pred-horizon', default=4, type=int)
@click.option('--num-epochs', default=600, type=int)
@click.option('--batch-size', default=500, type=int)
@click.option('--lr', default=4e-3, type=float) #4e-3 4e-4 
@click.option('--ntrajs', default=8192, type=int)
@click.option('--uscale', default=10.0, type=float)
@click.option('--logdir', default='simplelog', type=str)
def run(seed, dt, pred_horizon, num_epochs, batch_size, lr, ntrajs, uscale, logdir):
    metrics = train(seed, dt, pred_horizon, num_epochs, batch_size, lr, ntrajs, uscale,
                    logdir)
    print(metrics)


if __name__ == '__main__':
    run()
