# 扩展卡尔曼滤波，时间步长动态变化
from matplotlib import pyplot as plt
import numpy as np
import h5py
from scipy.signal import savgol_filter

class MultiJointEKF:
    def __init__(self, n_joints=10, base_dt=0.001):
        self.n_joints = n_joints
        self.base_dt = base_dt
        self.state_dim = 3  # [q, v, a]
        
        # 观测矩阵（q, v, a 三通道观测）
        self.H = np.eye(3)  # 3x3矩阵
        
        # 初始噪声参数（可动态调整）
        self.Q_base = np.diag([1e-6, 1e-6, 5e-4])  # 过程噪声基准
        self.R = np.kron(np.eye(n_joints), np.diag([1e-6, 1e-6, 2e-2]))  # 观测噪声
        
        # 初始化状态矩阵（n_joints x 3）
        self.x = np.zeros((n_joints, self.state_dim))
        
        # 协方差矩阵（分块对角）
        self.P = np.kron(np.eye(n_joints), np.eye(self.state_dim)*0.1)
        
        self.last_time = None  # 记录上次时间戳

    def predict(self, current_time):
        """动态时间步长预测"""
        # 确保时间差为标量[7](@ref)
        if self.last_time is None:
            dt = float(self.base_dt)
        else:
            dt = float(current_time - self.last_time)  # 转换为标量
            
        # 修复F矩阵初始化问题[6](@ref)
        F = np.array([
            [1.0, dt, 0.5*dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)  # 统一数据类型
        
        # 动态调整过程噪声
        Q_scale = dt / self.base_dt
        self.Q = np.kron(np.eye(self.n_joints), 
                        self.Q_base * np.diag([Q_scale**3, Q_scale, Q_scale]))
        
        # 分块状态转移矩阵
        F_block = np.kron(np.eye(self.n_joints), F)
        
        # 状态预测（保持矩阵形状）
        self.x = (F_block @ self.x.reshape(-1, 1)).reshape(self.n_joints, self.state_dim)
        self.P = F_block @ self.P @ F_block.T + self.Q
        
        self.last_time = current_time
        return self.x

    def update(self, z):
        """三通道观测更新"""
        H_block = np.kron(np.eye(self.n_joints), self.H)
        
        # 残差计算（添加维度检查）
        if z.shape != (self.n_joints, 3):
            raise ValueError(f"观测矩阵维度错误，期望({self.n_joints},3)，实际{z.shape}")
            
        residual = z.reshape(-1, 1) - H_block @ self.x.reshape(-1, 1)
        
        # 卡尔曼增益（添加数值稳定性处理）
        S = H_block @ self.P @ H_block.T + self.R
        try:
            K = self.P @ H_block.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H_block.T @ np.linalg.pinv(S)  # 使用伪逆增强稳定性
            
        # 状态更新
        correction = (K @ residual).reshape(self.n_joints, self.state_dim)
        self.x += correction
        
        # 协方差更新（添加数值稳定性处理）
        I = np.eye(self.n_joints * self.state_dim)
        self.P = (I - K @ H_block) @ self.P
        return self.x

def get_dataset(h5_file_path):
    """
    从HDF5文件中读取数据，包括时间戳、关节位置 q、关节速度 dq 和驱动力 tau，
    并利用关节速度差分计算加速度 ddq。
    为了滤波后各数组维度一致，对加速度数据在第一个时刻进行填充。
    """
    if h5_file_path is None:
        print('Please input h5 file path!')
        return None
    else:
        with h5py.File(h5_file_path, 'r') as f:
            timestamps = f['timestamps'][:]
            q = f['q'][:].squeeze(axis=1)
            dq = f['dq'][:].squeeze(axis=1)
            tau = f['tau'][:].squeeze(axis=1)
        
        # 计算连续时刻的时间差
        time_diff = timestamps[1:] - timestamps[:-1]
        # 标记时间差小于 0.001 秒的异常点
        invalid_indices = np.where(time_diff < 0.001)[0] + 1  
        mask = np.ones(len(timestamps), dtype=bool)
        mask[invalid_indices] = False

        timestamps_cpu = timestamps[mask]
        q_cpu = q[mask]
        dq_cpu = dq[mask]
        tau_cpu = tau[mask]
        
        # 重新计算时间差用于加速度计算
        time_diff_valid = timestamps_cpu[1:] - timestamps_cpu[:-1]
        dq_diff = dq_cpu[1:] - dq_cpu[:-1]
        ddq = dq_diff / time_diff_valid
        
        # 为使 ddq 与其他数据维度一致，在第一个时刻填入第一个计算值（也可以设为 0）
        ddq = np.vstack((ddq, ddq[-1, :]))
        # print(ddq[-10:, 2])
        return timestamps_cpu, q_cpu, dq_cpu, ddq

if __name__ == "__main__":
    # 读取数据集（添加异常处理）
    try:
        timestamps, q_obs, v_obs, a_obs = get_dataset('sim2realmujocodataset.h5')
    except Exception as e:
        print(f"dataload failed: {str(e)}")
        exit()
    
    # 初始化EKF（添加参数校验）
    ekf = MultiJointEKF(n_joints=10)
    accelerations = np.zeros_like(a_obs)
    q_filtered = np.zeros_like(q_obs)
    v_filtered = np.zeros_like(v_obs)
    
    # 运行滤波器（添加进度提示）
    total_frames = len(timestamps)-1
    for i in range(total_frames):
        current_time = timestamps[i+1]
        
        # 预测阶段（添加时间有效性检查）
        if not np.isfinite(current_time):
            print(f"unvalid timestemp at index {i+1}")
            continue
            
        ekf.predict(current_time)
        
        # 构造观测矩阵（添加NaN检查）
        z = np.column_stack((q_obs[i], v_obs[i], a_obs[i]))
        if np.any(np.isnan(z)):
            print(f"NaN on the H index {i}")
            z = np.nan_to_num(z)
            
        # 更新阶段
        state = ekf.update(z)
        
        # 记录状态
        q_filtered[i] = state[:, 0]
        v_filtered[i] = state[:, 1]
        accelerations[i] = state[:, 2]
    
    # 可视化（添加多关节显示）
    plt.figure(figsize=(15,8))
    for j in range(3):  # 显示前3个关节
        plt.subplot(3,1,j+1)
        # plt.plot(v_obs[:,j], 'b-', alpha=0.6, label='v_obs')
        # plt.plot(v_filtered[:,j], 'g--', label='v_filtered')
        plt.plot(a_obs[:,j], 'g-', alpha=0.6, label='acc')
        plt.plot(accelerations[:,j], 'r--', label='filted acc')
        plt.title(f'joint {j} acc compare')
        plt.legend()
    plt.tight_layout()
    plt.show()