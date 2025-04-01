import pandas as pd
import matplotlib.pyplot as plt

# 数据读取与预处理 
df1 = pd.read_csv('simplelog/isaacdynamic202503251616/progress.csv') #simplelog/20250323_221240/progress.csv
df2 = pd.read_csv('simplelog/sim2realmujoco20250326_095628/progress.csv')
df3 = pd.read_csv('simplelog/20250331_234321/progress.csv')
df4 = pd.read_csv('simplelog/20250331_142704/progress.csv')

# df2 = pd.read_csv('simplelog/20250330_230712/progress.csv')
# df3 = pd.read_csv('simplelog/20250331_093001/progress.csv')
# df['training/epochs'] = df['training/epochs'].fillna(0)  # 处理缺失的epoch值

# 设置中文显示（如需要）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # Windows/Mac中文支持
# plt.rcParams['axes.unicode_minus'] = False
fig, ax1 = plt.subplots(figsize=(12, 6))

# 主坐标轴：训练损失
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('q Loss', color=color)
ax1.plot(df1['training/loss/q_loss/mean'], 
        color='lightblue', linestyle='--', label='training/loss/q_loss/mean')
ax1.plot(df2['training/loss/q_loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/q_loss/mean')
ax1.plot(df3['training/loss/q_loss/mean'], 
        color='lightblue', linestyle='-.', label='training/loss/q_loss/mean')
ax1.plot(df4['training/loss/q_loss/mean'], 
        color='lightblue', linestyle=':', label='training/loss/q_loss/mean')
ax1.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))


fig, ax2 = plt.subplots(figsize=(12, 6))
# 主坐标轴：训练损失
color = 'tab:blue'
ax2.set_xlabel('Epochs')
ax2.set_ylabel('q Loss', color=color)
ax2.plot(df1['training/loss/v_loss/mean'], 
        color='lightblue', linestyle='--', label='training/loss/v_loss/mean')
ax2.plot(df2['training/loss/v_loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/v_loss/mean')
ax2.plot(df3['training/loss/v_loss/mean'], 
        color='lightblue', linestyle='-.', label='training/loss/v_loss/mean')
ax2.plot(df4['training/loss/v_loss/mean'], 
        color='lightblue', linestyle=':', label='training/loss/v_loss/mean')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

fig, ax3 = plt.subplots(figsize=(12, 6))
# 主坐标轴：训练损失
color = 'tab:blue'
ax3.set_xlabel('Epochs')
ax3.set_ylabel('q Loss', color=color)
ax3.plot(df1['training/loss/u_loss/mean'], 
        color='lightblue', linestyle='--', label='training/loss/u_loss/mean')
ax3.plot(df2['training/loss/u_loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/u_loss/mean')
ax3.plot(df3['training/loss/u_loss/mean'], 
        color='lightblue', linestyle='-.', label='training/loss/u_loss/mean')
ax3.plot(df4['training/loss/u_loss/mean'], 
        color='lightblue', linestyle=':', label='training/loss/u_loss/mean')
ax3.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

fig, ax4 = plt.subplots(figsize=(12, 6))
# 主坐标轴：训练损失
color = 'tab:blue'
ax4.set_xlabel('Epochs')
ax4.set_ylabel('q Loss', color=color)
ax4.plot(df3['training/loss/qddot_loss/mean'], 
        color='lightblue', linestyle='-.', label='training/loss/qddot_loss/mean')
ax4.plot(df4['training/loss/qddot_loss/mean'], 
        color='lightblue', linestyle=':', label='training/loss/qddot_loss/mean')
ax4.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# 图表装饰
plt.title('plot[3,6](@ref)')

plt.grid(alpha=0.3)

plt.show()