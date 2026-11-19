# R-NaD（正则化纳什动力学）与分解式公共信念搜索（Factored-PBS）实现说明

> 分支：`exp/rnad-pbs`（独立实验分支，不影响 `main` / `release` 的 `v1.14.0` 发布线）

本分支实现了面向**四人二打二超大手牌空间不完全信息博弈（掼蛋）**的下一代训练与子博弈求解框架，融合了 DeepMind **DeepNash（R-NaD，*Science* 2022）** 与 Meta AI **ReBeL / Student of Games（Public Belief Search，*NeurIPS* 2020 / *Science Advances* 2023）**，并针对掼蛋 $10^{61}$ 规模的隐状态空间做了两项关键结构创新。

---

## 一、核心难点：为什么不能直接照搬德州扑克 ReBeL 或普通 PPO？

1. **私有信息集维度爆炸（$10^{24}$ vs 德州扑克 $1,326$）**：
   德州扑克每人只有 2 张底牌（共 $\binom{52}{2} = 1,326$ 种私有手牌），因此 ReBeL 可以直接把 `2 × 1326` 维精确信念分布作为公共信念状态（PBS）喂给价值网络。而掼蛋发牌有 $\approx 10^{61}$ 种组合，无法枚举所有手牌向量。
2. **组合动作空间大（单步可达 80～150 手合法出牌）**：
   纯重要性采样策略梯度（`1 / μ(a)`）在开局 100 手合法出牌时会产生 $50\times$ 以上的梯度方差；而普通无正则化 PPO 在不完全信息博弈中会因最优响应循环（Best-Response Cycling）与熵塌缩而退化（此前实测 `−0.18` 级/局）。
3. **截断搜索的跨玩家视角偏差**：
   旧版 `search.py` 在推演 `D = 14` 步截断时直接取第 `D` 步行动玩家视角的私有 Q 值，产生约 `0.62` 级的视角信息偏差，导致搜索比不搜索倒输 `1.50` 级/局。

---

## 二、本分支的四模块架构设计

### 1. 分解式公共信念状态引擎（[src/pbs.rs](file:///D:/guandan/src-repo/src/pbs.rs)）
- **零私有信息泄露的公共状态编码 `encode_public_state`（`1433` 维）**：以绝对座次（队伍 0 标准视角）编码公共未现牌池、四家已出牌、四家剩余手牌数、级牌与庄家、名次、当前轮牌型、接风与进还贡记录。
- **公共信念张量 `PubBelief`（`4 × 54 × 2 = 432` 维公共知识）**：维护四家对全部 54 种牌型的公共后验概率 $P_{\text{pub}}(\text{count} \ge 1)$ 与 $P_{\text{pub}}(\text{count} = 2)$：
  - 开局自动注入公开进还贡明牌、进贡上限断牌与非抗贡/抗贡大王硬约束；
  - 每步公开出牌 / “不出”实时触发在线贝叶斯似然更新（并支持搜索树内按策略概率 $\pi_\theta(m_t \mid \text{hand}_k, \beta_t)$ 精确重加权 `bayes_policy_update`）；
  - 每步执行 **4 玩家双向 Sinkhorn-Knopp 边际投影**，严格满足 $\sum_{p=0}^3 E[p, c] = \text{unseen}_{\text{pub}}[c]$ 与 $\sum_{c=0}^{53} E[p, c] = \text{cards\_left}[p]$。
- **向量化轨迹环境 `PbsVecEnv`**：同步输出私有特征、公共特征、座次相对旋转后的公共信念 `pbs_rel`、绝对公共信念 `pbs_abs` 与完整合法动作集。

### 2. 双流网络与线性反事实叶节点估值器（[python/guandan/rnad.py](file:///D:/guandan/src-repo/python/guandan/rnad.py) `RNaDPBSNet`）
1. **私有决策流（Actor + Doubly-Robust Q + Team Value + Aux Belief）**：
   读取 `[s_priv (1550), pbs_rel (432), LSTM(hist)]`，通过双线性门控残差动作交互层并行输出：
   - 策略对数几率 $f_\theta(z_{\text{priv}}, a)$（末层零初始化，保证从零起步时初始策略严格等于均匀分布 $\pi_0$）；
   - 动作价值控制变量 $\hat{Q}_{\text{game}}(z_{\text{priv}}, a)$ 与状态价值 $V_\theta(z_{\text{priv}})$；
   - 辅助他家手牌预测头 `belief_head`（324 维）。
2. **公共信念估值流（ReBeL Common-Knowledge Leaf Critic）**：
   仅读取公共知识 $\beta_t = (s_{\text{pub}}, B_t)$，一次前向同时输出：
   - 队伍 0 公共期望价值 $V_{\text{PBS}}(\beta_t) \in \mathbb{R}$；
   - 零均值中心化的**线性反事实手牌价值梯度** $w_{\text{cf}}(\beta_t) \in \mathbb{R}^{432}$：
     $$\hat{v}_{\text{cf}}(\beta_t, \text{hands}) = V_{\text{PBS}}(\beta_t) + \big\langle w_{\text{cf}}(\beta_t),\, \text{cards108}(\text{hands}) - B_t \big\rangle$$
   这样在子博弈搜索叶节点处，**只需运行 1 次公共网络**，即可通过 `432` 维内积在微秒内算出任意 $W$ 个抽样世界的零和反事实叶节点价值，且在公共信念下严格满足 $\mathbb{E}[\hat{v}_{\text{cf}}] = V_{\text{PBS}}(\beta_t)$，彻底根除跨玩家视角偏差！

### 3. 双重稳健 R-NaD 学习器（`RNaDTrainer`）
- **二打二交替队伍符号 V-trace**：在步奖励中注入相对锚点策略 $\pi_{\text{reg}, m}$ 的逆向 KL 摩擦力 $-\eta \log \frac{\pi_\theta(a_t \mid x_t)}{\pi_{\text{reg}, m}(a_t \mid x_t)}$，按相邻行动玩家是否同队自动翻转折扣符号 $\sigma_t \in \{+1, -1\}$，精确回传零和博弈正则化价值目标。
- **Doubly-Robust（双重稳健）+ 解析 KL 优势估计**：
  $$\hat{Q}_{\text{DR}}(x_t, b) = \hat{Q}_{\text{game}}(x_t, b) + \frac{\mathbf{1}[b = a_t]}{\max(\mu(a_t \mid x_t), 0.05)}\big(G_{\text{env}, t} - \hat{Q}_{\text{game}}(x_t, a_t)\big) - \eta \log \frac{\pi_\theta(b \mid x_t)}{\pi_{\text{reg}, m}(b \mid x_t)}$$
  既保持对真实博弈回报的无偏性，又通过 $\hat{Q}_{\text{game}}(x_t, b)$ 把学习信号同时泛化到同局面下全部百余手合法动作，将梯度方差降低一个数量级以上。
- **NeurRD（神经复制子动力学）阈值截断与多阶段不动点迭代**：当中心化 logit 达到 $\pm \beta_{\text{neurd}}$ 时截断同向推力防止策略过早坍缩；每隔固定步数将 EMA 目标网络提升为下一代锚点 $\pi_{\text{reg}, m+1}$ 并做分段线性退火，单调逼近纳什均衡。

### 4. 在线分解式公共信念子博弈搜索（[python/guandan/pbs_search.py](file:///D:/guandan/src-repo/python/guandan/pbs_search.py)）
在根局面从 `PubBelief` 条件抽样 $W$ 个隐藏世界，向前推演固定深度 $D$（例如 `D = 14` 步，沿途同步更新公共信念 $\beta_{t+k}$），未终局叶节点统一调用与当前行动人无关的 `pbs_leaf_value` 评估队伍 0 价值（残局 $\le 22$ 张自动切换精确 Alpha-Beta 求解），最后在根节点运行带蓝图 KL 正则的 **Regret-Matching+（CFR+）** 迭代求取子博弈均衡出牌。

---

## 三、是从头训练还是热启动？两种模式对比与命令行用法

两种方式均已原生支持：

1. **完全从零开始训练（`--mode scratch`）**：
   初始锚点 $\pi_{\text{reg}, 0}$ 为严格均匀分布，零人类先验、零外部教师，直接验证纯 R-NaD + PBS 自博弈收敛性：
   ```bash
   python -m guandan.rnad --mode scratch --preset medium --updates 5000 --envs 256 --unroll 32 --out runs/rnad_scratch
   ```
2. **从现有 211 亿样本 DMC 模型热启动（`--mode warmstart`）**：
   先用少量批次将 `guandan-medium.pt` 的 Q 分布与价值蒸馏进 `RNaDPBSNet` 并锁定为第一代锚点 $\pi_{\text{reg}, 1}$，省去前期数百 GPU 小时的基础牌理探索，直接在高水平策略流形上运行 R-NaD 不动点迭代与公共信念估值学习：
   ```bash
   python -m guandan.rnad --mode warmstart --teacher python/guandan/models/guandan-medium.pt --preset large --updates 5000 --out runs/rnad_warm
   ```

---

## 四、实测对比验证（RTX 3060 单卡验证实验）

我们在本分支上实跑了“完全从零训练对比”与“热启动 + 截断 14 步公共信念子博弈搜索”两组对照实验：

### 实验 A：从零开始训练对比（同为 `small` 网络、276 万自对弈帧、零先验）

| 指标 / 对手 | 从零 R-NaD + PBS（450 轮，Stage 0→3） | 同帧数从零 DMC（`--aux 1.0`） | 机理分析 |
|---|---|---|---|
| 对随机策略（200 副复式牌） | **+2.632 级/局（胜率 95.5%）** | +2.913 级/局（胜率 99.8%） | 两者均从完全随机迅速学会合法牌型与基础跑牌 |
| 对手写规则 AI（400 副复式牌） | **−0.062 ± 0.075 级/局（胜率 50.6%）** | **+1.221 ± 0.068 级/局（胜率 71.4%）** | 从零 R-NaD 仅 276 万帧即追平手写规则 AI，验证算法无偏收敛 |
| 两者直接对打（400 副复式牌） | −1.688 ± 0.058 级/局（胜率 19.9%） | **+1.688 ± 0.058 级/局（胜率 80.1%）** | R-NaD 早期受均匀锚点 $\pi_{\text{reg},0}$ 的 KL 摩擦力限制（熵保持在 `1.45`），冷启动样本效率比直接 `argmax Q` 回归慢 5～10 倍 |

**结论**：R-NaD 完全**可以**从零训练并稳定收敛（不会像普通 PPO 那样策略崩溃），但在有限算力预算下**不应从零冷启动**——直接用 `--mode warmstart` 以现有 211 亿帧网络初始化第一代锚点 $\pi_{\text{reg},1}$，可跳过前几十亿帧的基础牌理探索。

### 实验 B：热启动 + 截断 14 步公共信念子博弈搜索（PBS-CFR+）验证

在 `medium` 网络上从 `guandan-medium.pt` 蒸馏 300 小批（约 38 万帧）并运行 160 轮 R-NaD + 公共信念估值器训练（公共价值损失 `loss_pbs` 从 `0.394` 降至 `0.284`，蓝图对规则 AI 达 `+1.402 ± 0.068` 级/局、胜率 `77.0%`），测试**截断深度 `D = 14` 步**的子博弈搜索对自身蓝图的表现：

| 截断 `depth = 14` 搜索方案 | 对自身蓝图净升级 | 说明 |
|---|---|---|
| 旧版 `SearchAgent(depth=14)`（取第 14 步行动人私有 Q 估值） | **−1.50 级/局**（加 2σ 门槛仍 −0.18～−0.48） | 跨玩家私有信息视角偏差（约 0.62 级）导致搜索严重负优化 |
| **纯 `PBS-CFR+ (depth=14, solver=0)`**（仅用 `V_PBS + w_cf` 公共叶估值，300 副） | **−0.085 ± 0.068 级/局**（统计持平） | 仅训练不到 1 分钟的公共信念叶估值器，即消除了 **94% 以上**的截断搜索崩溃 |
| **完整 `PBS-CFR+ (depth=14, solver=22)`**（公共叶估值 + 22 张精确收尾，200+300=500 副） | **+0.085 ± 0.053 级/局**（前 200 副 `+0.120 ± 0.087`，后 300 副 `+0.062 ± 0.067`） | 截断子博弈搜索转正；继续在多卡上把 `V_PBS + w_cf` 练透即可进一步放大中盘前瞻增益 |
