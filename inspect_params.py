"""
第6周 域随机化 · 第一步:探查 HalfCheetah 的物理参数。
先别急着改,先看清楚我们能动哪些"材料"、它们长什么样。
"""

import gymnasium as gym
import numpy as np

# 创建环境(不需要画面,我们只是来读参数的)
env = gym.make("HalfCheetah-v5")

# env.unwrapped.model 就是 MuJoCo 的物理模型对象
# (注意:这个 model 是物理模型,不是 PPO 那个训练用的 model!)
model = env.unwrapped.model

print("=" * 50)
print("HalfCheetah 物理参数探查")
print("=" * 50)

# ── 1. 各部位质量 ────────────────────────────
# body_mass 是个数组,每个元素是一个身体部位的质量
print("\n【各部位质量 body_mass】")
print(f"一共有 {len(model.body_mass)} 个部位")
print(f"质量数组:{np.round(model.body_mass, 3)}")

# ── 2. 摩擦力 ────────────────────────────────
# geom_friction 是个二维数组,每行对应一个几何面
# 每行有3个值,第一个是主要的滑动摩擦系数
print("\n【摩擦力 geom_friction】")
print(f"一共有 {len(model.geom_friction)} 个接触面")
print(f"摩擦力数组(每行第一个数是主滑动摩擦):")
print(np.round(model.geom_friction, 3))

print("\n" + "=" * 50)
print("这些就是我们待会儿要随机化的'材料'。")
print("=" * 50)

env.close()