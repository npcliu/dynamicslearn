import h5py
import numpy as np
import matplotlib.pyplot as plt

class MultiJointKalmanFilter:
    def __init__(self, n_joints=10, dt=0.001):
        self.n_joints = n_joints
        self.dt = dt
        self.state_dim = 3  # [q, v, a] per joint
        
        # 单关节状态转移矩阵[8](@ref)
        self.F = np.array([
            [1, dt, 0.5*dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # 观测矩阵 (仅观测q和v)[3](@ref)
        self.H = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 2x3
        
        # 噪声协方差矩阵（分块对角）[7](@ref)
        self.Q = np.kron(np.eye(n_joints), np.diag([1e-4, 1e-4, 1e-3]))  # 过程噪声
        self.R = np.kron(np.eye(n_joints), np.diag([1e-4, 1e-4, 1e-2]))         # 观测噪声
        
        # 初始化状态矩阵（n_joints x 3）
        self.x = np.zeros((n_joints, self.state_dim))
        
        # 协方差矩阵（分块对角）[5](@ref)
        self.P = np.kron(np.eye(n_joints), np.eye(self.state_dim)*0.1)

    def predict(self):
        """预测阶段（分块矩阵运算）[7](@ref)"""
        # 构建分块状态转移矩阵
        F_block = np.kron(np.eye(self.n_joints), self.F)
        
        # 状态预测 (保持矩阵形状)
        self.x = (F_block @ self.x.reshape(-1, 1)).reshape(self.n_joints, self.state_dim)
        
        # 协方差预测
        self.P = F_block @ self.P @ F_block.T + self.Q
        return self.x

    def update(self, z):
        """更新阶段（z为n_joints x 2的观测矩阵）[3](@ref)"""
        # 构建分块观测矩阵
        H_block = np.kron(np.eye(self.n_joints), self.H)  # (2n_joints x 3n_joints)
        
        # 计算残差（保持列向量形式）
        residual = z.reshape(-1, 1) - H_block @ self.x.reshape(-1, 1)
        
        # 计算卡尔曼增益[8](@ref)
        S = H_block @ self.P @ H_block.T + self.R
        K = self.P @ H_block.T @ np.linalg.inv(S)  # (3n_joints x 2n_joints)
        
        # 状态更新（保持矩阵形状）
        correction = (K @ residual).reshape(self.n_joints, self.state_dim)
        self.x += correction
        
        # 协方差更新[7](@ref)
        I = np.eye(self.n_joints * self.state_dim)
        self.P = (I - K @ H_block) @ self.P
        return self.x

def generate_test_data(n_joints=10, duration=9, dt=0.001):
    """生成多关节正弦运动测试数据[3,6](@ref)"""
    t = np.arange(0, duration, dt)
    n_samples = len(t)
    
    # 生成各关节运动参数（不同频率和振幅）
    freqs = np.linspace(1, 5, n_joints)  # 1-5Hz
    amps = np.linspace(0.5, 2.0, n_joints)  # 0.5-2.0m
    
    # 真实运动模型
    q_true = np.array([amp * np.sin(2*np.pi*freq*t) for amp, freq in zip(amps, freqs)]).T
    v_true = np.array([amp*2*np.pi*freq * np.cos(2*np.pi*freq*t) for amp, freq in zip(amps, freqs)]).T
    a_true = np.array([-amp*(2*np.pi*freq)**2 * np.sin(2*np.pi*freq*t) for amp, freq in zip(amps, freqs)]).T

    # 添加传感器噪声
    q_noise = np.random.normal(0, 0.05, q_true.shape)  # 位置噪声1cm
    v_noise = np.random.normal(0, 0.5, v_true.shape)   # 速度噪声10cm/s
    a_noise = np.random.normal(0, 10, v_true.shape)   # 速度噪声10cm/s
    return t, q_true + q_noise, v_true + v_noise, a_true + a_noise

def evaluate_performance(a_true, a_est, joint_idx=0):
    """性能评估与可视化[3,6](@ref)"""
    # 计算指标
    rmse = np.sqrt(np.mean((a_est[:, joint_idx] - a_true[:, joint_idx])**2))
    max_error = np.max(np.abs(a_est[:, joint_idx] - a_true[:, joint_idx]))
    
    # 可视化对比
    plt.figure(figsize=(12, 6))
    plt.plot(a_true[500:1500, joint_idx], 'g-', label='True Acceleration')
    plt.plot(a_est[500:1500, joint_idx], 'r--', label='KF Estimation')
    plt.title(f'Joint {joint_idx} Acceleration Estimation\nRMSE: {rmse:.4f}, Max Error: {max_error:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_dataset(h5_file_path):
    if h5_file_path==None:
        print('please input h5 file!!!')
    else:
        with h5py.File(h5_file_path, 'r') as f:
            timestamps = f['timestamps'][:]
            q = f['q'][:].squeeze(axis=1)
            dq = f['dq'][:].squeeze(axis=1)
            tau = f['tau'][:].squeeze(axis=1)
            
        # 时间差计算（向量化运算优化）
        time_diff = timestamps[1:] - timestamps[:-1]
        # print(time_diff[0:10])
        # print(min(time_diff))
        invalid_indices = np.where(time_diff < 0.001)[0] + 1  # 标记异常点
        
        # 构建掩码数组
        mask = np.ones(len(timestamps), dtype=bool)
        mask[invalid_indices] = False

        # 数据过滤 过滤掉时间差值小于1毫秒的数据
        timestamps_cpu = timestamps[mask]
        # time_diff = timestamps_cpu[1:] - timestamps_cpu[:-1]
        # print(min(time_diff))
        q_cpu = q[mask]
        dq_cpu = dq[mask]
        tau_cpu = tau[mask]
        
        time_diff = timestamps_cpu[1:] - timestamps_cpu[:-1]
        dq_diff = dq_cpu[1:] - dq_cpu[:-1]
        ddq = dq_diff / time_diff

        return timestamps_cpu, q_cpu, dq_cpu, ddq
        
# 使用示例
if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子
    t, q_obs, v_obs, a_obs = get_dataset('sim2realmujocodataset.h5')
    # 生成测试数据
    # t, q_obs, v_obs, a_obs = generate_test_data()
    
    # 初始化滤波器
    kf = MultiJointKalmanFilter(n_joints=10, dt=0.001)
    accelerations = np.zeros_like(a_obs)
    q = np.zeros_like(a_obs)
    v = np.zeros_like(a_obs)
    # 运行滤波器
    for i in range(len(t)):
        kf.predict()
        z = np.column_stack((q_obs[i], v_obs[i], a_obs[i]))
        state = kf.update(z)
        accelerations[i] = state[:, 2]
        q[i] = state[:, 0]
        v[i] = state[:, 1]
    # 评估第3关节性能
    evaluate_performance(a_obs, accelerations, joint_idx=3)
    
    # 打印各关节RMSE
    for j in range(10):
        rmse = np.sqrt(np.mean((accelerations[:, j] - a_obs[:, j])**2))
        print(f"Joint {j}: RMSE = {rmse:.6f}")