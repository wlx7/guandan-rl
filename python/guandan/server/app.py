"""FastAPI application: static web UI, REST API for training / rooms, WebSocket for play."""
from __future__ import annotations

import io
import os
import socket
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..paths import list_models, resolve_model, default_model, home, set_model_entry
from . import trainer
from . import perm
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

STATE = {"rooms": {}, "port": 8000}

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


_own = {"t": 0.0, "ips": ()}


def own_ips():
    """This computer's own addresses: opening the LAN address on the host still counts as the host."""
    if time.time() - _own["t"] > 60:
        _own["ips"], _own["t"] = tuple(lan_ips()), time.time()
    return _own["ips"]


def ident(conn) -> dict:
    """Who is asking (an HTTP request or a WebSocket): host or guest, and the browser's device id."""
    host = conn.client.host if conn.client else None
    token = conn.headers.get("x-guandan-admin") or conn.query_params.get("admin")
    device = (conn.headers.get("x-guandan-device") or conn.query_params.get("device") or "")[:64]
    return {"admin": perm.is_admin(host, token, own_ips()), "local": perm.is_local(host, own_ips()),
            "device": device, "host": host, "token": token}


def need_admin(request: Request, what: str):
    if not ident(request)["admin"]:
        raise HTTPException(403, f"只有房主可以{what}")


def need_train(request: Request):
    if not (ident(request)["admin"] or perm.load()["guest_train"]):
        raise HTTPException(403, "训练中心只有房主可以使用")


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


@app.get("/api/me")
def api_me(request: Request):
    i = ident(request)
    st = perm.load()
    return {"admin": i["admin"], "local": i["local"], "pin_set": bool(st["pin_hash"]),
            "guest_create": st["guest_create"], "guest_train": st["guest_train"]}


@app.post("/api/perm/login")
def api_perm_login(body: dict, request: Request):
    try:
        token = perm.login(str(body.get("pin") or ""), request.client.host if request.client else "?")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"token": token}


@app.post("/api/perm/logout")
def api_perm_logout(request: Request):
    perm.logout(ident(request)["token"])
    return {"ok": True}


@app.post("/api/perm/settings")
def api_perm_settings(body: dict, request: Request):
    need_admin(request, "修改权限设置")
    perm.update(body.get("guest_create"), body.get("guest_train"))
    if "pin" in body:
        pin = str(body.get("pin") or "").strip()
        if pin and len(pin) < 4:
            raise HTTPException(400, "房主密码至少 4 位")
        perm.set_pin(pin)
    return api_me(request)


# screen sizes the browsers report (to fit the table layout to the phones actually used)
CLIENTS: list[dict] = []


@app.post("/api/client")
def api_client(body: dict, request: Request):
    i = ident(request)
    row = {"t": time.strftime("%H:%M:%S"), "host": i["host"], "device": i["device"][:8]}
    for k in ("w", "h", "sw", "sh", "dpr", "fullscreen", "ua"):
        v = body.get(k)
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 200):
            row[k] = v
    CLIENTS.append(row)
    del CLIENTS[:-40]
    return {"ok": True}


@app.get("/api/clients")
def api_clients(request: Request):
    need_admin(request, "查看设备屏幕信息")
    return {"clients": CLIENTS[::-1]}


@app.get("/api/info")
def api_info():
    try:
        from ..model import preset_table, RECOMMENDED_PRESET
        presets = preset_table()
    except ImportError:  # portable build: no PyTorch, play only
        presets, RECOMMENDED_PRESET = [], "medium"
    ips = lan_ips()
    port = STATE["port"]
    return {
        "version": __version__,
        "lan_urls": [f"http://{ip}:{port}/" for ip in ips],
        "home": home(),
        "device": device_info(),
        "presets": presets,
        "can_train": trainer.can_train(),
        "portable": trainer.PORTABLE,
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
def api_models_manage(body: dict, request: Request):
    """Rename and/or hide a model-list entry: {"key": "runs/<name>" | "models/<file>", "label"?, "hidden"?}."""
    need_admin(request, "整理模型列表")
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
def api_train_start(opts: dict, request: Request):
    need_train(request)
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
def api_train_stop(body: dict, request: Request):
    need_train(request)
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
def api_train_export(body: dict, request: Request):
    need_train(request)
    try:
        p = trainer.export_model(body["run"], body["file"], body.get("as") or body["run"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"path": p}


# ------------------------------------------------------------------ rooms
# Several tables can run at once (each with its own players, AI and progress); the table list
# on the home page manages them.
MAX_ROOMS = 8


def rooms() -> list[Room]:
    """Open tables, oldest first."""
    for k in [k for k, r in STATE["rooms"].items() if r.closed]:
        del STATE["rooms"][k]
    return list(STATE["rooms"].values())


def get_room(room_id: str | None = None, token: str | None = None) -> Room | None:
    """The table with this id, the table that holds this seat token, or (neither given) the
    newest table, so that old clients and links keep working."""
    rs = rooms()
    if token:
        for r in rs:
            if r.seat_of_token(token) is not None:
                return r
    if room_id:
        return STATE["rooms"].get(room_id) if room_id in STATE["rooms"] and not STATE["rooms"][room_id].closed else None
    return rs[-1] if rs else None


def need_manage(r: Room, request: Request):
    """The host, or the device that opened this table."""
    if not r.can_manage(ident(request)):
        raise HTTPException(403, "只有房主或开这张桌的人可以管理这张牌桌")


@app.get("/api/rooms")
def api_rooms(request: Request):
    i = ident(request)
    return {"rooms": [dict(r.summary(), can_manage=r.can_manage(i)) for r in rooms()], "max": MAX_ROOMS}


@app.get("/api/room")
def api_room(request: Request, id: str = ""):
    r = get_room(id or None)
    if r is None:
        return {"exists": False}
    return dict(r.public_info(), exists=True, can_manage=r.can_manage(ident(request)))


@app.post("/api/room/create")
def api_room_create(body: dict, request: Request):
    me = ident(request)
    if not (me["admin"] or perm.load()["guest_create"]):
        raise HTTPException(403, "房主没有开放其他设备开桌，请让房主开桌后扫码加入")
    mode = body.get("mode", "teach")
    seats_in = body.get("seats") or []
    model_default = resolve_model(body.get("model") or "") or default_model()
    if model_default is None:
        raise HTTPException(400, "没有可用的模型，请先训练或下载模型")
    if len(rooms()) >= MAX_ROOMS:
        raise HTTPException(400, f"最多同时开 {MAX_ROOMS} 桌，请先在首页关掉不用的牌桌")
    seats = []
    for i in range(4):
        s = seats_in[i] if i < len(seats_in) else {"kind": "ai"}
        kind = s.get("kind", "ai")
        if mode == "teach":
            kind = "human" if i == 0 else "ai"
        m = resolve_model(s.get("model") or "") or model_default
        seats.append({"kind": kind, "model": m if kind == "ai" else None, "name": s.get("name", "")})
    room = Room(mode, seats, hint_model=model_default, timeout=int(body.get("timeout", 20)),
                ai_speed=float(body.get("ai_speed", 1.0)),
                length="quick" if body.get("length") == "quick" else "full",
                think="deep" if body.get("think") == "deep" else "fast",
                scoring_name=body.get("scoring") or "none")
    room.owner = me["device"]
    STATE["rooms"][room.id] = room
    token = None
    seat = None
    # host_plays = False: this computer only hosts (and shows the table); the human seats are
    # taken by phones or other devices on the LAN
    if room.human_seats() and body.get("host_plays", True):
        seat, token = room.join(0 if seats[0]["kind"] == "human" else None, body.get("name") or "房主")
    return {"room": room.id, "seat": seat, "token": token}


@app.post("/api/room/join")
def api_room_join(body: dict):
    r = get_room(body.get("room"), body.get("token"))
    if r is None:
        if body.get("room") and body.get("token"):
            raise HTTPException(410, "这张牌桌已经关闭了")
        raise HTTPException(404, "这张牌桌不存在或已经关闭")
    try:
        seat, token = r.join(body.get("seat"), body.get("name", ""), body.get("token"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"room": r.id, "seat": seat, "token": token}


@app.post("/api/room/fill_ai")
async def api_room_fill_ai(request: Request, body: dict | None = None):
    r = get_room((body or {}).get("room"))
    if r is None or r.status != "waiting":
        raise HTTPException(400, "牌桌已经开始或不存在")
    need_manage(r, request)
    r.fill_ai()
    r.maybe_start()
    await r.broadcast()
    return {"ok": True}


@app.post("/api/room/pause")
async def api_room_pause(body: dict, request: Request):
    """Pause or resume a table from the table list."""
    r = get_room(body.get("room"))
    if r is None:
        raise HTTPException(404, "这张牌桌不存在或已经关闭")
    need_manage(r, request)
    r.set_pause(bool(body.get("paused")), "房主")
    await r.broadcast()
    return {"ok": True}


@app.post("/api/room/close")
async def api_room_close(request: Request, body: dict | None = None):
    rid = (body or {}).get("room")
    r = get_room(rid) if rid else None
    if r is not None:
        need_manage(r, request)
    elif not rid:
        need_admin(request, "关闭所有牌桌")
    targets = [r] if r is not None else ([] if rid else rooms())  # no id: close all (old clients)
    for x in targets:
        await x.end("房主关闭了这张牌桌")
        STATE["rooms"].pop(x.id, None)
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    token = ws.query_params.get("token")
    rid = ws.query_params.get("room")
    r = get_room(rid, token)
    if r is None:
        await ws.send_json({"type": "error", "message": "这张牌桌已经关闭了" if (rid or token) else "当前没有牌桌", "fatal": True})
        await ws.close()
        return
    r.idents[ws] = ident(ws)
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
        r.idents.pop(ws, None)
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
