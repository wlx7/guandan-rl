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
from ..paths import list_models, resolve_model, default_model, home
from . import trainer
from .room import Room

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="Guandan RL", version=__version__)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

STATE = {"room": None, "port": 8000}


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.")]

    def score(ip):  # prefer typical home Wi-Fi ranges
        return (0 if ip.startswith("192.168.") else 1 if ip.startswith("10.") else 2, ip)
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
def api_models():
    out = []
    for m in list_models():
        out.append({k: m[k] for k in ("id", "source", "mtime", "size")})
    return out


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
        trainer.stop_run(body["name"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


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
                length="quick" if body.get("length") == "quick" else "full")
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
