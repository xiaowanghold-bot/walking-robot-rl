"""
第二个强化学习项目:训练 HalfCheetah(猎豹)学会向前奔跑。
这是连续控制任务 —— 智能体要同时控制 6 个关节的力矩。
"""

import gymnasium as gym
from stable_baselines3 import PPO

# ── 1. 创建环境 ──────────────────────────────
# 和 CartPole 唯一的区别:环境名字换成了 "HalfCheetah-v5"。
# 注意:这里我们不开画面(不加 render_mode),训练才快。
env = gym.make("HalfCheetah-v5")

# ── 2. 创建 PPO 智能体 ───────────────────────
# 和 CartPole 完全一样的一行!
# PPO 会自动识别出这是连续动作空间,内部切换成"输出钟形分布"的模式 ——
# 你不用做任何额外的事,这就是框架替你处理好的。
model = PPO("MlpPolicy", env, verbose=1)

# ── 3. 训练 ─────────────────────────────────
# 关键区别:连续控制比平衡杆难得多,需要的训练量大很多。
# CartPole 只要 2 万步;HalfCheetah 这里给 30 万步打底。
# 有 GPU,这大概需要几分钟到十几分钟,耐心等。
print("开始训练 HalfCheetah...")
model.learn(total_timesteps=300000)
print("训练完成!")

# ── 4. 保存大脑 ─────────────────────────────
model.save("ppo_halfcheetah")
print("模型已保存为 ppo_halfcheetah.zip")

# ── 5. 简单测试:看它现在每局能跑多远(累计奖励)─────
# 注意:这里的分数不再是"坚持步数",而是累计奖励 ——
# 大致反映"跑得多快 + 多省力"。分数越高越好,
# 训练好的 HalfCheetah 通常能到几千分。
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