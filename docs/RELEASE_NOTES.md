## 掼蛋 AI

纯自对弈强化学习训练的掼蛋 AI，带训练中心与局域网对战。

### Windows 一键安装

1. 下载下面的 `install_windows.bat`、`install_windows.ps1`、`start_windows.bat`、`train_windows.bat`，放进同一个文件夹。
2. 双击 `install_windows.bat`：自动安装 Python 3.12、CUDA 版 PyTorch（自动匹配显卡驱动）和本程序。
3. 双击 `start_windows.bat` 打开界面。局域网里的朋友用手机扫牌桌上的二维码即可加入。

Linux / macOS：下载对应平台的 `.whl`，`pip install torch guandan_rl-*.whl`，然后运行 `guandan serve`。

### 内容

- 完整掼蛋规则：逢人配、钢板、三连对、同花顺、天王炸、进贡 / 还贡 / 抗贡、接风、双上升三级、过 A、打 A 三次不过退回 2
- Rust 规则引擎 + PyTorch 深度蒙特卡洛自对弈训练，模型大小可选（推荐 medium：2.46M 参数，fp16 约 5 MB）
- 训练中心：从零或从基准模型训练，实时胜率曲线（对规则 AI / 基准模型 / 随机）
- 教学模式（AI 提示、出牌点评、明牌）与实战模式（20 秒限时，超时随机出牌，出完可看队友牌）
- 快速对局（随机级牌）与竞技模式（打到 A）
- `guandan-medium.pt`：随包附带的训练模型（安装后自动可用）
