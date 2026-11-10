# 安卓版

安卓版是一个 App（`guandan_android.apk`），在手机上运行和电脑版一样的牌桌：和 AI 打、教学模式的提示与点评、快速对局和竞技模式、计分倍数、多桌、权限都一样，界面也是同一套。它不能训练（训练需要 PyTorch 和显卡），用的是随包模型。

这台手机同时是局域网主机：同一 WiFi 下的其他手机或电脑扫牌桌上的二维码，用浏览器就能加入，不用装 App。

## 安装

1. 在手机上打开 [Releases](https://github.com/wlx7/guandan-rl/releases)，下载 `guandan_android.apk`。
2. 点下载好的文件安装。第一次会提示“禁止安装未知来源应用”，按提示允许浏览器（或文件管理器）安装应用即可。部分手机还会提示“未经检测的应用”，选“继续安装”。
3. 打开“掼蛋 AI”。第一次会问能否发送通知，建议允许：通知里显示局域网地址，也可以从通知里退出。

以后有新版本，下载新的 APK 直接安装即可覆盖升级，牌桌设置和房主密码都会保留。

## 使用

- App 固定横屏全屏，打开就是首页，和电脑版一样点“开一桌”。
- 想和朋友一起打：大家连同一个 WiFi，在牌桌上点二维码按钮，朋友用手机相机或微信扫码即可。也可以让朋友在浏览器里输入通知栏或首页显示的地址（例如 `http://192.168.1.5:8000/`）。
- 切到别的 App 时，只要还有别人在牌桌上，牌局就继续（你的座位由 AI 托管）；没人在时牌局自动暂停，回来接着打。
- 在后台超过 10 分钟且没有人在任何牌桌上，App 会自动退出以省电。也可以在通知里点“退出”立即结束（所有牌桌随之关闭）。
- 返回键：在牌桌上是返回上一页；在首页会把 App 切到后台（不会关掉牌桌）。

## 权限说明

| 权限 | 用途 |
|---|---|
| 网络、WiFi 状态 | 运行牌桌服务，让同一 WiFi 的设备连进来，显示局域网地址 |
| 前台服务、通知 | 切到后台时保持牌桌运行，并在通知里显示地址和“退出” |
| 保持唤醒 | 别人还在打牌时，锁屏后服务器也不睡眠 |

App 不访问网络上的任何服务器，不读取通讯录、位置、照片等任何个人数据。

## 技术说明

- `android/server`：牌桌服务器的 Rust 版。规则引擎就是根目录的 `guandan_core`；房间逻辑、计时、托管、提示与点评、权限和 HTTP / WebSocket 接口照搬 `python/guandan/server`，消息格式完全相同，所以 `python/guandan/web` 的网页界面原样使用（编译时嵌入）。神经网络推理也用 Rust 实现，和 PyTorch / numpy 的结果误差在 1e-6 以内，算一步出牌在电脑上约 2 毫秒，手机上也远快于 AI 的出牌动画。
- 模型：`packaging/export_gdm.py` 把随包的 `guandan-medium.pt` 转成 safetensors 格式（fp16，约 11 MB）放进 APK。
- `android/app`：Java 写的外壳。`MainActivity` 是全屏横屏的 WebView（用户代理带 `GuandanApp`，网页据此隐藏“全屏”按钮）；`ServerService` 是前台服务，持有 WiFi 锁和唤醒锁，显示通知。
- 电脑上也能运行同一个服务器，方便调试：

  ```bash
  python packaging/export_gdm.py python/guandan/models/guandan-medium.pt guandan-medium.gdm
  cd android/server && cargo run --release -- --model ../../guandan-medium.gdm --port 8000
  ```

- CI（`.github/workflows/build.yml` 的 `android` 与 `android-test`）：先在 Linux 上用同一份代码跑一桌 AI 对局，再用 cargo-ndk 编译 arm64-v8a 和 x86_64 两种 CPU 的库、用 Gradle 打包 APK，最后在安卓模拟器里安装、打开 App，通过它的服务器打一桌 AI 对局。
- 签名：CI 的 `ANDROID_KEYSTORE_B64`（keystore 的 base64）和 `ANDROID_KEYSTORE_PASSWORD` 两个 secret 设置后用正式密钥签名；没有设置时用仓库里公开的测试密钥 `android/app/debug.keystore`，保证每个版本都能覆盖安装上一个版本。换成正式密钥后需要先卸载一次旧版。
