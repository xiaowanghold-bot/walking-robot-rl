"""
观看成果:加载训练好的模型,弹出窗口实时看 AI 把杆子立起来。
"""

import gymnasium as gym
from stable_baselines3 import PPO

# ── 1. 创建带画面的环境 ────────────────────────
# 关键区别:render_mode="human" 表示要弹出一个可视化窗口。
# (训练时我们没开画面,是为了跑得快;现在要看,所以打开。)
env = gym.make("Walker2d-v5", render_mode="human")

# ── 2. 加载你刚训练好的大脑 ──────────────────────
# 读取上一步保存的 ppo_walker2d.zip。
model = PPO.load("ppo_walker2d")

# ── 3. 让它表演 3 局,你盯着窗口看 ─────────────────
for episode in range(3):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"第 {episode + 1} 局:坚持了 {int(total_reward)} 步")

env.close()
print("表演结束!")