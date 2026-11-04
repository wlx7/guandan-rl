#!/usr/bin/env bash
# 掼蛋 AI：在 Linux GPU 云主机（AutoDL、恒源云等）上一键安装并训练。
#
# 用法（在云主机的终端里）：
#   bash cloud_train.sh bench          # 约 10 分钟：测几种并行设置的训练速度，和笔记本对比
#   bash cloud_train.sh train [小时]    # 后台长时间训练（默认 10 小时），关掉网页也不会停
#   bash cloud_train.sh status         # 查看训练进度
#   bash cloud_train.sh stop           # 保存并停止训练
#   bash cloud_train.sh pack           # 把最新模型复制到 export/，方便下载回本机
#
# 可选环境变量：ACTORS（自对弈进程数，默认用 bench 测出的最快值，否则 3）、BASE（起点模型路径）
set -e
MODE=${1:-bench}
HERE=$(cd "$(dirname "$0")" && pwd)
# AutoDL 自带的“学术加速”（访问 GitHub 更快）；其他平台没有这个文件就跳过
if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1 || true; fi
if [ -d /root/autodl-tmp ]; then ROOT=/root/autodl-tmp/guandan; else ROOT=${GUANDAN_DIR:-$HOME/guandan}; fi
mkdir -p "$ROOT"
cd "$ROOT"
export GUANDAN_HOME="$ROOT/guandan_data"
PY="$ROOT/.venv/bin/python"

install() {
  if [ ! -x "$PY" ]; then
    # reuse the image's CUDA PyTorch when it has one
    python3 -m venv --system-site-packages .venv
  fi
  if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "安装 PyTorch ..."
    "$PY" -m pip install -q torch
  elif command -v nvidia-smi >/dev/null 2>&1 && ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "镜像里的 PyTorch 不支持显卡，重新安装 CUDA 版 ..."
    "$PY" -m pip install -q --upgrade torch
  fi
  # the program itself: a wheel next to this script (uploaded by hand) wins; otherwise GitHub
  local url
  url=$(ls -t "$HERE"/guandan_rl-*manylinux*.whl "$ROOT"/guandan_rl-*manylinux*.whl 2>/dev/null | head -1 || true)
  if [ -z "$url" ]; then
    url=$("$PY" - <<'EOF' || true
import json, urllib.request
r = json.load(urllib.request.urlopen("https://api.github.com/repos/wlx7/guandan-rl/releases/latest", timeout=20))
print(next(a["browser_download_url"] for a in r["assets"] if "manylinux" in a["name"] and a["name"].endswith(".whl")))
EOF
)
  fi
  if [ -z "$url" ]; then
    echo "连不上 GitHub。请把 guandan_rl-*-manylinux*.whl 上传到和本脚本相同的文件夹，再运行一次。"
    exit 1
  fi
  local have want
  have=$("$PY" -c "import guandan; print(guandan.__version__)" 2>/dev/null || echo none)
  want=$(basename "$url" | sed -E 's/guandan_rl-([0-9.]+)-.*/\1/')
  if [ "$have" != "$want" ]; then
    echo "安装 guandan-rl $want ..."
    "$PY" -m pip install -q --force-reinstall --no-deps "$url"
    "$PY" -m pip install -q numpy
  fi
  "$PY" -c "import guandan; print('guandan-rl', guandan.__version__)"
  "$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无（只能用 CPU）')"
  echo "CPU 核数: $(nproc)"
}

base_model() {
  if [ -n "$BASE" ]; then echo "$BASE"; return; fi
  "$PY" -c "import os; from guandan.paths import builtin_models_dir; print(os.path.join(builtin_models_dir(), 'guandan-medium.pt'))"
}

fps_of() {  # average self-play samples per second over the last minutes of a run
  "$PY" - "$GUANDAN_HOME/runs/$1/metrics.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tr = [r for r in rows if r.get("type") == "train"]
tr = tr[len(tr) // 3:]
print(round(sum(r["fps"] for r in tr) / max(1, len(tr))) if tr else 0, round(100 * sum(r.get("learner_busy", 0) for r in tr) / max(1, len(tr))))
EOF
}

case "$MODE" in
  bench)
    install
    B=$(base_model)
    CPU=$(nproc)
    best=0; best_a=3
    for A in 3 6 9 12; do
      if [ "$A" -gt $((CPU - 2)) ] && [ "$A" -ne 3 ]; then break; fi
      name="bench-a$A"
      rm -rf "$GUANDAN_HOME/runs/$name"
      echo "== 测速：$A 个自对弈进程，约 3 分钟 =="
      "$PY" -m guandan.train --name "$name" --base "$B" --actors "$A" --max-hours "${BENCH_HOURS:-0.05}" --eval-interval 100000 > "$ROOT/$name.log" 2>&1 || true
      read -r fps busy < <(fps_of "$name")
      echo "   每秒 $fps 个样本，学习进程忙碌 $busy%"
      if [ "${fps:-0}" -gt "$best" ]; then best=$fps; best_a=$A; fi
    done
    echo "$best_a" > "$ROOT/best_actors"
    echo
    echo "最快：$best_a 个自对弈进程，每秒约 $best 个样本（参考：RTX 3060 笔记本约 3.4 万）。"
    echo "约是笔记本的 $(awk "BEGIN{printf \"%.1f\", $best/34000}") 倍。开始长时间训练：bash cloud_train.sh train 10"
    ;;
  train)
    install
    HOURS=${2:-10}
    A=${ACTORS:-$(cat "$ROOT/best_actors" 2>/dev/null || echo 3)}
    B=$(base_model)
    NAME=${NAME:-cloud}
    echo "后台训练 $HOURS 小时：从 $B 开始，$A 个自对弈进程，训练名 $NAME"
    nohup "$PY" -m guandan.train --name "$NAME" --base "$B" --actors "$A" --max-hours "$HOURS" \
      --eval-interval 900 --eval-rounds 400 > "$ROOT/$NAME.log" 2>&1 &
    sleep 60
    tail -3 "$ROOT/$NAME.log"
    echo "已在后台运行，关掉网页也不会停。查看进度：bash cloud_train.sh status"
    ;;
  status)
    if [ -f "$ROOT/bench/hill/summary.txt" ] && [ -z "$NAME" ]; then
      # automatic tuning (bench/hill_loop.sh) is in charge: show its progress
      echo "== 自动调参记录 =="
      tail -8 "$ROOT/bench/hill/summary.txt"
      cur=$(pgrep -af "guandan.train --name" | sed -E 's/.*--name ([^ ]+).*/\1/' | head -1)
      if [ -n "$cur" ]; then
        echo "== 正在训练：$cur =="
        grep -E "^\[eval\]" "$ROOT/bench/hill/$cur.log" | tail -2 || true
        tail -1 "$ROOT/bench/hill/$cur.log"
      else
        echo "（此刻没有训练在跑：可能在两轮之间做对比评估）"
      fi
      pgrep -f hill_loop.sh >/dev/null && echo "自动调参程序：运行中" || echo "自动调参程序：未运行"
    else
      NAME=${NAME:-cloud}
      grep -E "^\[eval\]" "$ROOT/$NAME.log" | tail -5 || true
      tail -2 "$ROOT/$NAME.log"
    fi
    ;;
  stop)
    if pgrep -f hill_loop.sh >/dev/null; then
      touch "$ROOT/bench/hill/STOP"
      cur=$(pgrep -af "guandan.train --name" | sed -E 's/.*--name ([^ ]+).*/\1/' | head -1)
      [ -n "$cur" ] && touch "$GUANDAN_HOME/runs/$cur/STOP"
      echo "已请求停止自动调参：当前训练约半分钟内保存后退出（以后从断点继续：删除 bench/hill/STOP 再运行 bench/hill_loop.sh）。"
    else
      NAME=${NAME:-cloud}
      touch "$GUANDAN_HOME/runs/$NAME/STOP"
      echo "已请求停止，约半分钟内保存模型后退出。"
    fi
    ;;
  pack)
    if [ -z "$NAME" ] && [ -f "$ROOT/bench/hill/state.json" ]; then
      NAME=$("$PY" -c "import json; print(json.load(open('$ROOT/bench/hill/state.json'))['best']['model'].split('/')[0])")
      echo "自动调参目前最强的模型：$NAME"
    fi
    NAME=${NAME:-cloud}
    mkdir -p "$ROOT/export"
    out="$ROOT/export/$NAME-$(date +%m%d-%H%M).pt"
    cp "$GUANDAN_HOME/runs/$NAME/model.pt" "$out"
    echo "模型已复制到：$out"
    echo "在网页的文件浏览器里下载它，放进本机 D:\\guandan\\guandan_data\\models\\ 就能在对战里选它。"
    ;;
  *)
    echo "用法：bash cloud_train.sh bench | train [小时] | status | stop | pack"; exit 1 ;;
esac
