import h5py
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
file_config = {
    'sim2sim': {
        'path': 'sim2realmujocodataset.h5', #sim2realmujocodataset sim2simmujocodataset
        'color': 'blue',
        'label': 'mujoco'
    },
    'isaac': {
        'path': 'isaacdataset.h5', 
        'color': 'red',
        'label': 'IsaacGym'
    }
}

# 创建画布
fig, axes = plt.subplots(4, 1, figsize=(12, 8), dpi=100)
fig.suptitle('Cross-Dataset Motion Comparison', fontsize=14)

# 数据缓存字典
dataset_cache = {}

# 遍历两个数据集
for dataset_key in ['sim2sim', 'isaac']:
    cfg = file_config[dataset_key]
    
    try:
        with h5py.File(cfg['path'], 'r') as file:
            # 读取关键数据
            dataset_cache[dataset_key] = {
                'timestamps': np.array(file['timestamps'][:]),
                'q_command': np.array(file['q_command'][:]),
                'root_states': np.array(file['root_states'][:]),
                'q': np.array(file['q'][:]),
                'root_v': np.array(file['root_v'][:]),
                'dq': np.array(file['dq'][:]),
                'tau': np.array(file['tau'][:])
            }
            
            print(f"[Success] Loaded {dataset_key}: {cfg['path']}")
            print(f"Data shape - q: {dataset_cache[dataset_key]['root_v'].shape}")
            
    except Exception as e:
        print(f"[Error] Failed to load {cfg['path']}: {str(e)}")
        continue

# 绘制对比曲线
time_ratio = 1.0  # 时间轴缩放系数（处理不同采样率）
for key in dataset_cache.keys():
    data = dataset_cache[key]
    
    # 关节角度对比（取第一个关节）
    axes[0].plot(
        data['timestamps'][:,0] * time_ratio, 
        data['q'][:,0,0], 
        color=file_config[key]['color'],
        linestyle='--',
        linewidth=1.2,
        alpha=0.7,
        label=f'{file_config[key]["label"]} - Joint 0'
    )
    
    # 关节力矩对比
    axes[1].plot(
        data['timestamps'][:,0] * time_ratio,
        data['dq'][:,0,0],
        color=file_config[key]['color'],
        linestyle='--', 
        linewidth=1.2,
        alpha=0.7,
        label=f'{file_config[key]["label"]} - Torque'
    )

    # 关节力矩对比
    axes[2].plot(
        data['timestamps'][:,0] * time_ratio,
        data['tau'][:,0,0],
        color=file_config[key]['color'],
        linestyle='--', 
        linewidth=1.2,
        alpha=0.7,
        label=f'{file_config[key]["label"]} - Torque'
    )
    
        # 关节力矩对比
    axes[3].plot(
        data['timestamps'][:,0] * time_ratio,
        data['q_command'][:,0,0],
        color=file_config[key]['color'],
        linestyle='--', 
        linewidth=1.2,
        alpha=0.7,
        label=f'{file_config[key]["label"]} - Torque'
    )
# 坐标轴设置
axes[0].set(
    xlabel='Time (s)',
    ylabel='Joint Position (rad)',
    title='Joint Angular Trajectory Comparison'
)
axes[0].grid(True, linestyle=':')
axes[0].legend(loc='upper right')

axes[1].set(
    xlabel='Time (s)', 
    ylabel='Joint velocity (m/s)',
    title='Joint velocity Profile Comparison'
) 
axes[1].grid(True, linestyle=':')
axes[1].legend(loc='upper right')

axes[2].set(
    xlabel='Time (s)', 
    ylabel='Joint Torque (Nm)',
    title='Joint Torque Profile Comparison'
) 
axes[2].grid(True, linestyle=':')
axes[2].legend(loc='upper right')

axes[3].set(
    xlabel='Time (s)', 
    ylabel='Joint command (rad)',
    title='Joint command Profile Comparison'
) 
axes[3].grid(True, linestyle=':')
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.show()