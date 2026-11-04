"""FastAPI application: static web UI, REST API for training / rooms, WebSocket for play."""
from __future__ import annotations

import io
import os
import socket
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..paths import list_models, resolve_model, default_model, home, set_model_entry
from . import trainer
from ..proc import resource_limits
from ..scoring import PRESETS as scoring_presets
from .room import Room

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="Guandan RL", version=__version__)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def no_stale_ui(request, call_next):
    # always revalidate the web UI so that an upgrade is picked up without a hard refresh
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

STATE = {"room": None, "port": 8000}

try:  # keep the battle server light: training (if any) needs the cores
    import torch as _torch
    _torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 4)))
except Exception:
    pass


def lan_ips():
    ips = set()
    primary = None
    try:  # the address of the interface that holds the default route (normally the Wi-Fi card)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.")]

    def score(ip):  # default-route address first, then typical home Wi-Fi ranges
        return (0 if ip == primary else 1 if ip.startswith("192.168.") else 2 if ip.startswith("10.") else 3, ip)
    if primary and not primary.startswith("127.") and primary not in ips:
        ips.append(primary)
    return sorted(ips, key=score)


def device_info():
    info = {"cuda": False, "gpu": None, "cpu_count": os.cpu_count(), "torch": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    return info


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/info")
def api_info():
    from ..model import preset_table, RECOMMENDED_PRESET
    ips = lan_ips()
    port = STATE["port"]
    return {
        "version": __version__,
        "lan_urls": [f"http://{ip}:{port}/" for ip in ips],
        "home": home(),
        "device": device_info(),
        "presets": preset_table(),
        "recommended_preset": RECOMMENDED_PRESET,
        "default_model": default_model(),
        "cpu_count": os.cpu_count(),
        "limits": {"cuda": resource_limits("cuda"), "cpu": resource_limits("cpu")},
        "scoring": {k: v for k, v in scoring_presets.items() if v},
        "active_runs": trainer.active_runs(),
    }


@app.get("/api/qr")
def api_qr(text: str):
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml")


@app.get("/api/models")
def api_models(all: bool = False):
    out = []
    for m in list_models(all=all):
        out.append({k: m[k] for k in ("id", "source", "group", "label", "hidden", "mtime", "size")})
    return out


@app.post("/api/models/manage")
def api_models_manage(body: dict):
    """Rename and/or hide a model-list entry: {"key": "runs/<name>" | "models/<file>", "label"?, "hidden"?}."""
    key = str(body.get("key") or "")
    if not (key.startswith("runs/") or key.startswith("models/") or key.startswith("builtin/")) or ".." in key:
        raise HTTPException(400, "无效的模型")
    if key.startswith("builtin/") and body.get("hidden"):
        raise HTTPException(400, "内置模型不能隐藏")
    set_model_entry(key, body.get("label"), body.get("hidden"))
    return {"ok": True}


# ------------------------------------------------------------------ training
@app.get("/api/train/runs")
def api_runs():
    return trainer.list_runs()


@app.post("/api/train/start")
def api_train_start(opts: dict):
    if opts.get("base"):
        p = resolve_model(opts["base"])
        if not p:
            raise HTTPException(400, "找不到基准模型")
        opts["base"] = p
    try:
        name = trainer.start_run(opts)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": name}


@app.post("/api/train/stop")
def api_train_stop(body: dict):
    try:
        if body.get("force"):
            trainer.force_stop(body["name"])
            return {"ok": True, "forced": True}
        r = trainer.stop_run(body["name"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return dict(r, ok=True)


@app.get("/api/train/metrics")
def api_train_metrics(name: str, since: float = 0.0):
    return {"rows": trainer.metrics(name, since), "status": trainer.run_status(name)}


@app.get("/api/train/log")
def api_train_log(name: str):
    return {"log": trainer.log_tail(name)}


@app.post("/api/train/export")
def api_train_export(body: dict):
    try:
        p = trainer.export_model(body["run"], body["file"], body.get("as") or body["run"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"path": p}


# ------------------------------------------------------------------ rooms
@app.get("/api/room")
def api_room():
    r: Room | None = STATE["room"]
    if r is None or r.closed:
        return {"exists": False}
    return dict(r.public_info(), exists=True)


@app.post("/api/room/create")
def api_room_create(body: dict):
    mode = body.get("mode", "teach")
    seats_in = body.get("seats") or []
    model_default = resolve_model(body.get("model") or "") or default_model()
    if model_default is None:
        raise HTTPException(400, "没有可用的模型，请先训练或下载模型")
    seats = []
    for i in range(4):
        s = seats_in[i] if i < len(seats_in) else {"kind": "ai"}
        kind = s.get("kind", "ai")
        if mode == "teach":
            kind = "human" if i == 0 else "ai"
        m = resolve_model(s.get("model") or "") or model_default
        seats.append({"kind": kind, "model": m if kind == "ai" else None, "name": s.get("name", "")})
    old: Room | None = STATE["room"]
    if old is not None:
        old.close()
    room = Room(mode, seats, hint_model=model_default, timeout=int(body.get("timeout", 20)),
                ai_speed=float(body.get("ai_speed", 1.0)),
                length="quick" if body.get("length") == "quick" else "full",
                think="deep" if body.get("think") == "deep" else "fast",
                scoring_name=body.get("scoring") or "none")
    STATE["room"] = room
    token = None
    seat = None
    if room.human_seats():
        seat, token = room.join(0 if seats[0]["kind"] == "human" else None, body.get("name") or "房主")
    return {"room": room.id, "seat": seat, "token": token}


@app.post("/api/room/join")
def api_room_join(body: dict):
    r: Room | None = STATE["room"]
    if r is None or r.closed:
        raise HTTPException(404, "当前没有牌桌")
    if body.get("room") and body["room"] != r.id and body.get("token"):
        raise HTTPException(410, "牌桌已经换了")
    try:
        seat, token = r.join(body.get("seat"), body.get("name", ""), body.get("token"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"room": r.id, "seat": seat, "token": token}


@app.post("/api/room/fill_ai")
async def api_room_fill_ai():
    r: Room | None = STATE["room"]
    if r is None or r.closed or r.status != "waiting":
        raise HTTPException(400, "牌桌已经开始或不存在")
    r.fill_ai()
    r.maybe_start()
    await r.broadcast()
    return {"ok": True}


@app.post("/api/room/close")
def api_room_close():
    r: Room | None = STATE["room"]
    if r is not None:
        r.close()
        STATE["room"] = None
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    token = ws.query_params.get("token")
    r: Room | None = STATE["room"]
    if r is None or r.closed:
        await ws.send_json({"type": "error", "message": "当前没有牌桌", "fatal": True})
        await ws.close()
        return
    seat = r.seat_of_token(token) if token else None
    if seat is None:
        if token:
            await ws.send_json({"type": "error", "message": "座位已失效，请重新加入", "fatal": True})
            await ws.close()
            return
        seat = -1
        r.spectators.add(ws)
    else:
        r.seats[seat].connections.add(ws)
    try:
        r.maybe_start()
        await r.broadcast()
        while True:
            msg = await ws.receive_json()
            if r.closed:
                await ws.send_json({"type": "error", "message": "牌桌已关闭", "fatal": True})
                break
            await r.handle(seat, ws, msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if seat >= 0:
            r.seats[seat].connections.discard(ws)
        else:
            r.spectators.discard(ws)
        try:
            await r.broadcast()
        except Exception:
            pass


@app.exception_handler(Exception)
async def on_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=500)
