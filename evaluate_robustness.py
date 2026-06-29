"""
第6周 域随机化 · 终极对决:
在一个两只猎豹都"没见过"的陌生极端环境里,比谁更扛得住。

选手A:温室版  ppo_halfcheetah        (只在完美世界练过)
选手B:大世面版 ppo_halfcheetah_domainrand (在多变世界练过)

测试环境:故意设成极端值 —— 地面很滑 + 猎豹整体加重。
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


# ── 一个"固定的陌生极端环境" ───────────────────────
# 注意:这次不是随机,而是固定一套极端参数,让两个模型在
# 完全相同的陌生环境下公平比较。
class HarshTestEnv(gym.Wrapper):
    def __init__(self, env, friction, mass_factor):
        super().__init__(env)
        self.friction = friction
        self.mass_factor = mass_factor
        self.original_mass = self.env.unwrapped.model.body_mass.copy()

    def reset(self, **kwargs):
        model = self.env.unwrapped.model
        # 整体加重(每个真实部位都乘同一个系数)
        for i in range(1, len(model.body_mass)):
            model.body_mass[i] = self.original_mass[i] * self.mass_factor
        # 设成很滑的地面
        model.geom_friction[:, 0] = self.friction
        return self.env.reset(**kwargs)


def evaluate(model_path, env, n_episodes=5):
    """加载一个模型,在给定环境里跑 n 局,返回平均累计奖励"""
    model = PPO.load(model_path)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    return np.mean(scores), scores


# ── 设定陌生极端环境:地面很滑(0.15)+ 猎豹加重(1.4倍)──
# 0.15 比域随机版训练时见过的最低(0.2)还低 → 对它也是"没见过"
FRICTION = 0.1
MASS_FACTOR = 1.2

print("=" * 55)
print(f"陌生极端环境:摩擦力={FRICTION}(很滑), 质量×{MASS_FACTOR}(加重)")
print("=" * 55)

# 选手A:温室版
env_a = HarshTestEnv(gym.make("Walker2d-v5"), FRICTION, MASS_FACTOR)
avg_a, scores_a = evaluate("ppo_walker2d", env_a)
env_a.close()

# 选手B:大世面版
env_b = HarshTestEnv(gym.make("Walker2d-v5"), FRICTION, MASS_FACTOR)
avg_b, scores_b = evaluate("ppo_walker2d_domainrand", env_b)
env_b.close()

# ── 公布结果 ─────────────────────────────────
print("\n【对决结果】")
print(f"温室版   (标准训练):  平均 {avg_a:7.1f}   各局 {np.round(scores_a, 0)}")
print(f"大世面版 (域随机化):  平均 {avg_b:7.1f}   各局 {np.round(scores_b, 0)}")
print("\n" + "=" * 55)
if avg_b > avg_a:
    print(f"✅ 域随机化胜出!在陌生环境里高出 {avg_b - avg_a:.1f} 分")
    print("   证明:见过大世面的策略,泛化能力更强、更鲁棒。")
else:
    print("⚠️ 这次温室版没输 —— 这也是有价值的结果,我们一起分析原因。")
print("=" * 55)