"""
第5周 实验:调超参数 —— 大学习率(放大10倍)。

控制变量:任务、奖励、训练步数全部不动,只把学习率从默认
0.0003 改成 0.003(大10倍),看训练会不会"步子迈太大"而震荡/变差。
"""

import gymnasium as gym
from stable_baselines3 import PPO

# 注意:用标准 HalfCheetah,不带任何奖励改造(这周只研究超参数)
env = gym.make("HalfCheetah-v5")

# ── 关键改动:在这里指定学习率 ────────────────────
# learning_rate=0.003 —— 比默认的 0.0003 大 10 倍。
# 其它参数全部保持默认,确保"只动这一个变量"。
model = PPO("MlpPolicy", env, learning_rate=0.00003, verbose=1)

print("开始训练(大学习率 0.00003)...")
model.learn(total_timesteps=300000)
print("训练完成!")

model.save("ppo_halfcheetah_lr_small")
print("模型已保存为 ppo_halfcheetah_lr_small.zip")

env.close()