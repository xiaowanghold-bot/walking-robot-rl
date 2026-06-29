"""
第6周 域随机化 · 核心:训练一个"见过大世面"的猎豹。

每局开始(reset)时,随机改变机器人各部位质量 + 地面摩擦力,
逼智能体学会"无论环境怎么变,我都能跑"的通用能力。
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


# ── 域随机化 Wrapper ─────────────────────────────
class DomainRandomizationWrapper(gym.Wrapper):
    """
    每次 reset 时,随机扰动物理参数:
      - 各部位质量:在默认值的 0.7~1.3 倍之间随机(跳过第0个world body)
      - 地面摩擦力:在 0.2~0.8 之间随机
    """
    def __init__(self, env):
        super().__init__(env)
        # 关键:开局先把"原始默认质量"存下来。
        # 因为我们每局都是"在默认值基础上乘随机系数",
        # 必须始终以原始值为基准,否则随机会在上一局的基础上累积、越跑越偏。
        self.original_mass = self.env.unwrapped.model.body_mass.copy()

    def reset(self, **kwargs):
        model = self.env.unwrapped.model

        # ① 随机化质量:每个部位各乘一个 0.7~1.3 的随机系数
        #    从第1个开始(下标0是world body,质量0,跳过)
        for i in range(1, len(model.body_mass)):
            factor = np.random.uniform(0.7, 1.3)
            model.body_mass[i] = self.original_mass[i] * factor

        # ② 随机化摩擦力:整体随机到 0.2~0.8(每行第一个数是主滑动摩擦)
        friction = np.random.uniform(0.2, 0.8)
        model.geom_friction[:, 0] = friction

        # 改完参数后,再正常 reset 开新局
        return self.env.reset(**kwargs)


# ── 用域随机化环境训练 ───────────────────────────
env = DomainRandomizationWrapper(gym.make("HalfCheetah-v5"))

model = PPO("MlpPolicy", env, verbose=1)

print("开始训练(域随机化版猎豹)...")
model.learn(total_timesteps=300000)
print("训练完成!")

model.save("ppo_halfcheetah_domainrand")
print("模型已保存为 ppo_halfcheetah_domainrand.zip")

env.close()