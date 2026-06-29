"""
第三个强化学习项目:训练 Walker2d(两足机器人)学会直立行走。
这是第一个"会摔倒"的任务 —— 智能体必须一边前进、一边保持平衡。
"""

import gymnasium as gym
from stable_baselines3 import PPO

# ── 1. 创建环境 ──────────────────────────────
# 还是熟悉的配方,只换了环境名:"Walker2d-v5"。
env = gym.make("Walker2d-v5")

# ── 2. 创建 PPO 智能体 ───────────────────────
# 又是一字未改的一行。17 维动作、会摔倒、要平衡 —— PPO 全自动应对。
model = PPO("MlpPolicy", env, verbose=1)

# ── 3. 训练 ─────────────────────────────────
# 平衡比奔跑难,所以训练量再加大:给 100 万步。
# 有 GPU 也要约十几分钟,挂着等,别急。
print("开始训练 Walker2d...")
model.learn(total_timesteps=1000000)
print("训练完成!")

# ── 4. 保存大脑 ─────────────────────────────
model.save("ppo_walker2d")
print("模型已保存为 ppo_walker2d.zip")

# ── 5. 测试:看它每局能拿多少累计奖励 ──────────────
# 分数 = 前进得分 + 存活奖励 - 能耗。
# 摔得早分就低;能稳稳走很久,分数会很高(训练好通常几千分)。
print("\n开始测试训练成果:")
for episode in range(3):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"  第 {episode + 1} 局:累计奖励 {total_reward:.0f}")

env.close()