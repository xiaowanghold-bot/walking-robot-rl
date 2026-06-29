# Walking Robot RL · 强化学习实践

> 从经典控制到机器人步态的强化学习学习项目。
> 当前进度:**阶段一 —— CartPole 平衡杆控制**(已完成)。

本仓库记录我系统学习强化学习(Reinforcement Learning)与具身智能的过程。目标是逐步从基础控制任务进阶到仿真环境中的机器人步态学习。

---

## 阶段一:CartPole 平衡杆

用 PPO(Proximal Policy Optimization)算法训练一个智能体,学会通过左右移动小车,让其上方的杆子尽可能长时间保持竖直不倒。

### 成果

训练 20,000 步后,智能体在测试中的表现(满分 500 步):

| 测试局 | 坚持步数 |
|:------:|:--------:|
| 第 1 局 | 500 |
| 第 2 局 | 409 |
| 第 3 局 | 462 |
| 第 4 局 | 397 |
| 第 5 局 | 487 |

加载最终模型可视化运行时,可稳定达到满分(500 步)。

> 训练全程在 NVIDIA GPU(WSL2 环境)下完成。

---

## 技术栈

- **语言**:Python 3.11
- **物理引擎 / 环境**:Gymnasium(CartPole-v1)
- **强化学习库**:Stable-Baselines3(PPO)
- **可视化**:pygame
- **开发环境**:WSL2 (Ubuntu) + Conda + VS Code

---

## 项目结构

```
walking-robot-rl/
├── train_cartpole.py    # 训练脚本:训练 PPO 智能体并保存模型
├── watch_cartpole.py    # 可视化脚本:加载模型,弹窗观看智能体表现
├── .gitignore           # 忽略模型文件与缓存
└── README.md
```

---

## 如何运行

### 1. 准备环境

```bash
# 创建并激活 conda 环境
conda create -n robot python=3.11 -y
conda activate robot

# 安装依赖
pip install mujoco gymnasium "stable-baselines3[extra]" pygame
```

### 2. 训练模型

```bash
python train_cartpole.py
```

运行后会训练智能体并生成模型文件 `ppo_cartpole.zip`。

### 3. 观看成果

```bash
python watch_cartpole.py
```

会弹出窗口,实时展示训练好的智能体平衡杆子的过程。

---

## 强化学习核心思路

智能体并非被编程"如何"平衡杆子,而是通过与环境的反复交互自己学会:

- **状态(State)**:小车位置、速度、杆子角度、角速度
- **动作(Action)**:向左 / 向右推小车
- **奖励(Reward)**:每多坚持一步 +1 分
- **学习**:PPO 算法根据奖励不断调整策略,逐步学会让杆子不倒

这套"感知状态 → 做出动作 → 获得反馈 → 改进策略"的循环,正是后续机器人步态学习的核心机制。

---

## 后续规划

- [x] 阶段二:连续控制任务(HalfCheetah / Hopper)
- [ ] 阶段三:双足 / 四足机器人步态学习(Walker2d / Ant)
- [ ] 阶段四:自定义任务与奖励设计、域随机化