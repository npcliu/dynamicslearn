import pandas as pd
import matplotlib.pyplot as plt

# 数据读取与预处理
df = pd.read_csv('/home/liu/bruce/brucesim2real/mechamodlearn/simplelog/seed=21,pred_horizon=10,batch_size=1000,lr=0.0001,ntrajs=8192,uscale=10.0,dt=0.001/progress.csv')
# df['training/epochs'] = df['training/epochs'].fillna(0)  # 处理缺失的epoch值

# 设置中文显示（如需要）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # Windows/Mac中文支持
# plt.rcParams['axes.unicode_minus'] = False
fig, ax1 = plt.subplots(figsize=(12, 6))

# 主坐标轴：训练损失
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('q Loss', color=color)
ax1.plot(df['training/loss/q_loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/q_loss/mean')

ax1.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))


fig, ax2 = plt.subplots(figsize=(12, 6))
# 主坐标轴：训练损失
color = 'tab:blue'
ax2.set_xlabel('Epochs')
ax2.set_ylabel('q Loss', color=color)
ax2.plot(df['training/loss/v_loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/v_loss/mean')

ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

fig, ax3 = plt.subplots(figsize=(12, 6))
# 主坐标轴：训练损失
color = 'tab:blue'
ax3.set_xlabel('Epochs')
ax3.set_ylabel('q Loss', color=color)
ax3.plot(df['training/loss/mean'], 
        color='lightblue', linestyle='-', label='training/loss/mean')

ax3.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
# # 副坐标轴：验证损失
# ax2 = ax1.twinx()  
# color = 'tab:red'
# ax2.set_ylabel('training/Validation/v Loss', color=color)  
# ax2.plot(df['training/loss/v_loss/mean']/7, 
#         color='lightblue', linestyle='--', label='training/loss/v_loss/mean')
# ax2.plot( df['training/loss/mean'], 
#         color='tab:blue', marker='s', label='training/loss/mean')
# ax2.plot(df['training/loss/mean'], 
#         color=color, marker='o', label='training/loss/mean')
# ax2.fill_between(
#                 df['training/loss/mean'] - df['training/loss/std'],
#                 df['training/loss/mean'] + df['training/loss/std'],
#                 color=color, alpha=0.1)
# ax2.tick_params(axis='y', labelcolor=color)

# 图表装饰
plt.title('plot[3,6](@ref)')

plt.grid(alpha=0.3)

plt.show()