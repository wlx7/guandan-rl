"""Entry point of the portable Windows build (one .exe, nothing to install).

Double-click: starts the battle server and opens the browser. It plays with the bundled
model in numpy (no PyTorch, so no training). Data (models you add, model names, records)
goes to a guandan_data folder next to the .exe, the same place the full install uses, so
the .exe put into the install folder sees the same models.

`guandan_windows.exe --selftest` plays one round with the bundled model and exits (CI).
"""
import json
import os
import socket
import sys
import urllib.request
import webbrowser


def exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.getcwd()


def running_guandan(port):
    """Version of a guandan server already listening on this port, or None."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/info", timeout=2) as r:
            return json.load(r).get("version") or "?"
    except Exception:
        return None


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def selftest():
    from guandan import _core, __version__
    from guandan.agents import GameAgent
    from guandan.paths import default_model
    agent = GameAgent(default_model())
    g = _core.Game(7)
    n = 0
    while g.phase not in (2, 3):
        k, key, cards = agent.choose(g)
        g.apply(k, key, list(cards))
        n += 1
    print(f"selftest ok: guandan {__version__}, numpy model {agent.numpy}, {n} moves, finished {list(g.finished)}")
    import fastapi, uvicorn, qrcode.image.svg  # noqa: F401  (the server's modules are bundled)
    from guandan.server import app  # noqa: F401
    return 0


def main():
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("GUANDAN_HOME", os.path.join(exe_dir(), "guandan_data"))
    args = sys.argv[1:]
    if "--selftest" in args:
        return selftest()
    port = 8000
    for a in args:
        if a.startswith("--port="):
            port = int(a.split("=", 1)[1])
    v = running_guandan(port)
    if v:
        print(f"掼蛋 AI（v{v}）已经在运行，直接打开浏览器。 / Guandan AI (v{v}) is already running; opening the browser.")
        if "--no-browser" not in args:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        return 0
    while not port_free(port) and port < 8100:
        port += 1
    from guandan.cli import main as cli
    return cli(["serve", "--port", str(port)] + (["--no-browser"] if "--no-browser" in args else []))


if __name__ == "__main__":
    try:
        code = main() or 0
    except SystemExit as e:
        code = e.code or 0
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    if code and getattr(sys, "frozen", False) and "--selftest" not in sys.argv:
        try:
            input("\n出错了（见上面的信息）。按回车键关闭窗口……\nSomething went wrong (see above). Press Enter to close this window...")
        except Exception:
            pass
    sys.exit(code)
