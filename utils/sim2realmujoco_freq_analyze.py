from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import h5py

def simple_amplitude_spectrum(data, title):
    fs = 1000  # 采样频率1kHz
    plt.figure(figsize=(15, 8))
    
    for joint in range(10):
        # 计算幅度谱
        f, Pxx = signal.welch(data[:, joint], fs, 
                            nperseg=4096,          # 优化频率分辨率
                            window='hann',
                            scaling='spectrum')
        
        # 绘制线性幅度谱
        plt.semilogy(f, Pxx, 
                    alpha=0.7,
                    label=f'Joint {joint+1}')  # 显示关节编号[7](@ref)

    # 图表美化设置
    plt.title(f'{title} Amplitude Spectrum', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude (log scale)', fontsize=12)
    plt.xlim(0, 500)                   # 聚焦有效频段[2](@ref)
    plt.grid(True, which='both', ls='--')  # 双网格系统[6](@ref)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    with h5py.File('sim2realmujocodataset.h5', 'r') as f:
        q_cpu = f['q'][:].squeeze(axis=1)
        dq_cpu = f['dq'][:].squeeze(axis=1)
        tau_cpu = f['tau'][:].squeeze(axis=1)
    
    simple_amplitude_spectrum(q_cpu, 'Joint Position')
    simple_amplitude_spectrum(dq_cpu, 'Joint Velocity')
    simple_amplitude_spectrum(tau_cpu, 'Joint Torque')