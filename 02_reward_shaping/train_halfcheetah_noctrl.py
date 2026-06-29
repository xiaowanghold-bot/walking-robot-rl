"""
第5周 实验一:改奖励函数 —— 拿掉能耗惩罚,看猎豹变成什么样。

我们用一个 Wrapper(包装器)把 HalfCheetah 的奖励改掉:
原本 reward = 前进奖励 - 能耗惩罚
改成 reward = 前进奖励 (能耗惩罚被抵消掉)

预测:猎豹会变得"不计代价、疯狂往前冲",动作更狂野。
"""

import gymnasium as gym
from stable_baselines3 import PPO


# ── 自定义 Wrapper:去掉能耗惩罚 ──────────────────
class NoCtrlCostWrapper(gym.Wrapper):
    """
    包装器:套在原 HalfCheetah 环境外面,拦截每一步的奖励并改写。
    gym.Wrapper 是 Gymnasium 提供的标准工具,继承它就能改造环境。
    """
    def step(self, action):
        # 1. 先让原环境正常走一步,拿到它原本的结果
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. info 里分开记着两部分奖励(第3周你见过 reward_ctrl)
        #    reward_ctrl 是负的(它是惩罚),例如 -0.3
        #    我们把它"减回去",就等于把这部分惩罚抵消掉:
        #    新reward = 原reward - reward_ctrl = 前进奖励 - 能耗 - (-能耗) = 前进奖励
        reward = reward - info["reward_ctrl"]

        # 3. 把改过的 reward 返回(其它原样不动)
        return obs, reward, terminated, truncated, info


# ── 用包装后的环境训练 ───────────────────────────
# 注意这一行:先 make 原环境,再用我们的 Wrapper 套起来
env = NoCtrlCostWrapper(gym.make("HalfCheetah-v5"))

model = PPO("MlpPolicy", env, verbose=1)

print("开始训练(无能耗惩罚版)...")
model.learn(total_timesteps=300000)
print("训练完成!")

model.save("ppo_halfcheetah_noctrl")
print("模型已保存为 ppo_halfcheetah_noctrl.zip")

env.close()