"""Command line entry point: `guandan serve | train | eval | info`."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser


def cmd_serve(args):
    import uvicorn
    from .server import app as appmod
    appmod.STATE["port"] = args.port
    ips = appmod.lan_ips()
    print("=" * 60)
    print(" 掼蛋 AI 已启动")
    print(f"   本机打开:   http://127.0.0.1:{args.port}/")
    for ip in ips:
        print(f"   局域网加入: http://{ip}:{args.port}/   (同一 WiFi 的手机/电脑)")
    print(f"   数据目录:   {os.path.abspath(__import__('guandan.paths', fromlist=['home']).home())}")
    print(" 关闭本窗口即可退出")
    print("=" * 60, flush=True)
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(f"http://127.0.0.1:{args.port}/")),
                         daemon=True).start()
    uvicorn.run(appmod.app, host=args.host, port=args.port, log_level="warning")


def cmd_train(args, rest):
    return subprocess.call([sys.executable, "-m", "guandan.train", *rest])


def cmd_eval(args, rest):
    return subprocess.call([sys.executable, "-m", "guandan.evaluate", *rest])


def cmd_info(args):
    from . import __version__
    from .model import preset_table
    from .paths import home, list_models
    print("guandan-rl", __version__)
    print("data dir:", home())
    try:
        import torch
        print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        print("torch not available:", e)
    print("presets:")
    for r in preset_table():
        print(f"  {r['name']:7s} params={r['params']:>9,}  fp16={r['size_fp16_mb']:5.1f}MB  "
              f"{'(推荐) ' if r['recommended'] else ''}{r['note']}")
    print("models:")
    for m in list_models():
        print("  ", m["id"])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="guandan", description="掼蛋 AI：训练与对战")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("serve", help="启动 Web 界面（训练中心 + 对战，支持局域网）")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--no-browser", action="store_true")
    sub.add_parser("train", help="命令行训练（参数同 python -m guandan.train）", add_help=False)
    sub.add_parser("eval", help="评估模型（参数同 python -m guandan.evaluate）", add_help=False)
    sub.add_parser("info", help="显示环境、模型尺寸预设与已有模型")
    if not argv:
        argv = ["serve"]
    if argv[0] in ("train", "eval"):
        return (cmd_train if argv[0] == "train" else cmd_eval)(None, argv[1:])
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "info":
        cmd_info(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
