# 与公开掼蛋 AI 的对战测评（2026-11）

本项目随包 AI（v1.11.x，large 网络 + 残局搜索）与网上能下载、能在 Linux 上运行的最强掼蛋 AI 对战。对局由**全国掼蛋 AI 算法大赛的官方离线裁判程序**判定，规则、发牌、进贡还贡都由它决定，我们的 AI 和对手一样，只是连上去的四个客户端之一。

## 结论

| 我方 vs | 小局数 | 小局胜率（95% 区间） | 平均每小局升级 | 整局胜率（从 2 打到过 A） |
|---|---|---|---|---|
| **DanZero**（AAAI 2023 官方模型） | 2,633 | **69.8%**（68.0–71.5%） | **+1.22 ± 0.04** | **274 / 280 = 97.9%** |
| DanZero，我方只用神经网络（不开残局搜索） | 2,184 | 70.1%（68.1–71.9%） | +1.25 ± 0.04 | 231 / 240 = 96.2% |
| **南邮掼蛋 AI**（首届全国赛，DanLM 修复版） | 1,780 | **90.7%**（89.3–92.0%） | **+2.35 ± 0.03** | **280 / 280 = 100%** |
| 对照：DanZero vs 南邮掼蛋 AI | 2,354 | 77.4%（75.6–79.0%） | +1.47 ± 0.04 | 275 / 280 = 98.2% |

- 对两个对手都是**明显优势**，不是持平。
- 优势几乎全部来自神经网络本身：关掉残局搜索后对 DanZero 的成绩不变（70.1% 对 69.8%，在误差范围内）。
- 座位没有影响：我方坐 0/2 号位和坐 1/3 号位，对 DanZero 的小局胜率分别是 69.9% 和 69.7%。

**和公开数字对照**（只能粗略比较，平台和规则实现不同）：DanLM 公布的单局胜率里，DanLM 对 DanZero 65.1%，DanZero V1T（增强复现版）对 DanZero 62.1%；DanZero 对南邮掼蛋 AI 71.4%，整局 95.5%。

## 对手

| 对手 | 来源 | 说明 |
|---|---|---|
| DanZero | [submit-paper/Danzero_plus](https://github.com/submit-paper/Danzero_plus) `wintest/danzero` | DanZero 论文（AAAI 2023）的 DMC 模型（567 维特征 → 5 × 512 → Q），160 个 CPU 训练 30 天。客户端代码原样运行，只把它向 TensorFlow 1 进程发请求的那一步换成同一个网络的 numpy 计算（`np.argmax(Q)`，与原 actor 相同）。 |
| 南邮掼蛋 AI | [dashidhy/DanLM](https://github.com/dashidhy/DanLM) `baselines/fin-njupt-guandan-ai` | 首届全国赛参赛程序（规则 AI），DanLM 修好了它导致约一半对局崩溃的 bug，并测得它是比赛程序里最强的一个。 |

**为什么 DanZero 的接入可信**：同一套环境里 DanZero 对南邮掼蛋 AI 赢了 77.4% 的小局、98.2% 的整局，不低于 DanLM 公布的 71.4% / 95.5%，说明它没有被接入方式削弱。

**没有测到的**（不是不想测，是拿不到能运行的模型）：

- DanLM（Botzone 掼蛋榜第一）：模型和引擎只有 macOS ARM 的编译文件，在 Linux 上无法运行。
- DanZero+（PPO 微调版）：仓库只提供 DMC 部分的权重，PPO 权重未公开。
- FableDan、DanKS（金山）：只开源训练代码，不提供训练好的权重。
- GS2、SDMC（OpenGuanDan 论文中的智能体）：没有公开模型。

## 方法

- **裁判**：Danzero_plus 仓库附带的全国赛离线平台 `danserver`（Linux 版）。一次“游戏”是一整局：从 2 打到某方过 A。
- **规模**：每个对手 14 张牌桌 × 20 局（只用网络的一组为 12 × 20），偶数桌我方坐 0/2 号位，奇数桌坐 1/3 号位。小局指一副牌（打到头游所在的一方确定升级数为止）。
- **硬件**：RunPod 32 核 CPU 机器（cpu5c），约 37 分钟。
- **我方如何接入**：裁判只告诉每个玩家自己的手牌和公开出牌。客户端（`packaging/arena/our_client.py`）用这些信息在本项目的引擎里重建牌局：自己的手牌是真的，其他三家用大小正确的占位牌，对手出牌前把那几张牌换进他的占位手牌（`Game.set_view` / `Game.ensure_holds`）。神经网络只读取自己的手牌和公开信息（见 `src/features.rs`），所以占位牌不会泄露任何信息；残局搜索本来就会重新抽样别人的手牌。
  - 用本项目引擎模拟裁判做的检验（`packaging/arena/fake_referee.py`）：近 5,000 次出牌决策里，重建出的网络输入与真实牌局**逐位相同**，选出的牌也完全相同。
  - 正式对局中每次决策都把重建的牌局与裁判公布的“当前最大牌”和各家剩余张数核对。

## 对我方不利的因素（结果没有因此偏向我方）

- **裁判规则与标准规则略有不同**，例如单张 A 有时不允许用级牌压，A-2-3-4-5 被当作大顺子（只有 10-J-Q-K-A 或炸弹能管）。约 1.6% 的决策里（对 DanZero：1,359 / 86,556），我方网络想出的牌裁判不允许，只能改出裁判允许的牌里网络评分最高的一手。DanZero 就是在这个平台上训练的，不受影响。
- 约 2,600 小局中有 5 局（10 次），裁判接受了一手按本项目规则管不上的牌，重建的牌局与裁判对不上；这几局剩下的部分改用不看出牌历史的简化方式出牌。
- 进贡阶段裁判只给出可选的最大牌，我方直接选第一张；双贡时裁判在两家都还贡后才通知，第二个还贡的人看不到第一家还了什么。

## 复现

```bash
# 在 Linux 机器上（需要 Rust、gcc、Python 3.10+）
pip install . websocket-client ws4py numpy
gcc -shared -fPIC -O2 -o packaging/arena/portshim.so packaging/arena/portshim.c -ldl
git clone --depth 1 https://github.com/submit-paper/Danzero_plus.git ext/Danzero_plus
git clone --depth 1 https://github.com/dashidhy/DanLM.git ext/DanLM
python -I packaging/arena/safe_unpickle.py ext/Danzero_plus/wintest/danzero/q_network.ckpt ext/danzero_q.npz
cp ext/Danzero_plus/wintest/torch/danserver ext/ && chmod +x ext/danserver

python packaging/arena/arena.py --opp danzero --tables 14 --games 20 --ext ext --server ext/danserver --out results/danzero
python packaging/arena/arena.py --opp njupt   --tables 14 --games 20 --ext ext --server ext/danserver --out results/njupt --base-port 25000
python packaging/arena/summarize.py results/danzero results/njupt
```

`safe_unpickle.py` 只接受纯 numpy 数组，DanZero 的权重文件里没有别的内容。裁判端口固定为 23456，`portshim.so` 把它换成每张桌各自的端口，才能在一台机器上同时开多桌。汇总数据见 [arena/summary.json](arena/summary.json)。

## English summary

Our bundled AI (large network + endgame search) played complete matches (level 2 through A) against the strongest GuanDan AIs that are downloadable and run on Linux, refereed by the official offline platform of the national GuanDan AI competition. Against **DanZero** (AAAI 2023, official weights) it won **69.8%** of 2,633 rounds (+1.22 levels per round) and **274 of 280** matches; with the network alone, 70.1% of rounds. Against the strongest bot of the 1st national competition (NJUPT, as fixed by DanLM) it won **90.7%** of rounds and **all 280** matches. As a check of the DanZero adapter, DanZero beat the NJUPT bot in 77.4% of rounds and 98.2% of matches here (DanLM reports 71.4% / 95.5%). DanLM, DanZero+ (PPO), FableDan and DanKS could not be included: their trained models are either not published or macOS-only. Referee rule quirks (about 1.6% of our decisions) and 5 desynchronised rounds can only have hurt our side.
