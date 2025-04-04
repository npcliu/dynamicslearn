import numpy as np
from mechamodlearn.rigidbody import LearnedRigidBody
import torch
from datetime import datetime
from pathlib import Path
from mechamodlearn import dataset, utils, viz_utils, rigidbody, models
import h5py

# DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
class ExternalData:
    def __init__(self, qdim, udim, device):

        self._qdim = qdim
        self._udim = udim
        self._device = device
        self.thetamask = torch.ones(10, device=DEVICE)
        print('self.thetamask',self.thetamask)
def get_dataset(system, T, dt, h5_file_path):
    t_points = torch.arange(0, T, dt, device=DEVICE).requires_grad_(True)
    if h5_file_path==None:
        timestamps_gpu_t = None
        dq_gpu_t = None
        tau_gpu_t = None
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
        
        len_t = len(t_points)
        N = q_gpu.shape[0]
        timestamps_gpu_t = torch.stack([timestamps_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        q_gpu_t = torch.stack([q_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        dq_gpu_t = torch.stack([dq_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        ddq_gpu_t = torch.stack([ddq_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0)
        tau_gpu_t = torch.stack([tau_gpu[i : i + (N - len_t + 1)] for i in range(len_t)], dim=0) 
        q_gpu_t = utils.wrap_to_pi(q_gpu_t.view(-1, system._qdim), system.thetamask).view(
            len_t, -1, system._qdim) 
        
        test_data = (timestamps_gpu_t[:, 50:52], q_gpu_t[:, 50:52, :], 
                        dq_gpu_t[:, 50:52, :], ddq_gpu_t[:, 50:52, :], tau_gpu_t[:, 50:52, :])
    test_dataset_ = dataset.ActuatedTrajectoryDataset.FromExternalData(test_data)
    # print(data.q_B_T.shape)
    return test_dataset_


if __name__ == '__main__':
    system = ExternalData(10, 10, DEVICE)
    T = 0.1
    dt = 0.001
    # test_dataset = get_dataset(system, T, dt, 'isaacdataset.h5')
    # model = torch.load('simplelog/20250325_161619/metrics_20250326-011730.pt', weights_only=False, map_location='cpu')
    test_dataset = get_dataset(system, T, dt, 'sim2realmujocodataset.h5')
    model = torch.load('simplelog/20250404_014428/full_model_epoch_599.pt', weights_only=False, map_location='cpu')
    
    # assert torch.allclose(model(test_input), loaded_model(test_input))
    t_points = torch.arange(0, T, dt, device=DEVICE)
    viz_utils.vizqvmodel(model,
                                test_dataset.q_B_T.to(device=DEVICE),
                                test_dataset.v_B_T.to(device=DEVICE),
                                test_dataset.ddq_B_T.to(device=DEVICE),
                                test_dataset.u_B_T.to(device=DEVICE), 
                                test_dataset.t_B_T.to(device=DEVICE), method='euler')