"""
第一个强化学习程序:训练 AI 玩 CartPole(平衡杆)
目标:让小车左右移动,把竖在上面的杆子尽量长时间地立住不倒。
"""

import gymnasium as gym
from stable_baselines3 import PPO

# ── 1. 创建环境 ──────────────────────────────
# "CartPole-v1" 是 Gymnasium 自带的经典环境:一辆小车,上面立一根杆子。
# 每撑住一步 +1 分,杆子倒了或小车跑太远就结束。满分 500。
env = gym.make("CartPole-v1")

# ── 2. 创建 PPO 智能体(它的"大脑")──────────────
# "MlpPolicy" 表示用一个普通的多层神经网络当大脑。
# verbose=1 让它训练时把进度打印出来,方便你观察。
model = PPO("MlpPolicy", env, verbose=1)

# ── 3. 训练 ─────────────────────────────────
# 让它和环境互动 20000 步,从一次次"杆子倒了"中学习怎么不倒。
print("开始训练...")
model.learn(total_timesteps=20000)
print("训练完成!")

# ── 4. 保存训练好的大脑 ──────────────────────────
model.save("ppo_cartpole")
print("模型已保存为 ppo_cartpole.zip")

# ── 5. 实战测试:看它现在能撑多少步 ─────────────────
# 跑 5 局,每局看它在杆子倒下前坚持了多少步(越接近 500 越好)。
print("\n开始测试训练成果:")
for episode in range(5):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        # deterministic=True:用它学到的最优策略,不再随机探索
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"  第 {episode + 1} 局:坚持了 {int(total_reward)} 步")

env.close()