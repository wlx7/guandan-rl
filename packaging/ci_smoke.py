"""CI check of the portable build: the server answers, and an all-AI table actually plays
over the WebSocket (bundled uvicorn/websockets, engine and numpy model all work)."""
import asyncio
import json
import sys
import time
import urllib.request

import websockets

BASE = "127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8000")


def get(path):
    with urllib.request.urlopen(f"http://{BASE}{path}", timeout=10) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(f"http://{BASE}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


for _ in range(90):
    try:
        info = get("/api/info")
        break
    except Exception:
        time.sleep(1)
else:
    sys.exit("server did not start")
print("info:", info["version"], "portable", info["portable"], "can_train", info["can_train"], info["default_model"])
assert info["portable"] and not info["can_train"] and info["default_model"]
assert get("/api/models"), "no models"
post("/api/room/create", {"mode": "real", "length": "quick", "ai_speed": 0.05, "seats": [{"kind": "ai"}] * 4})


async def watch():
    async with websockets.connect(f"ws://{BASE}/ws") as ws:
        first = None
        deadline = time.time() + 120
        while time.time() < deadline:
            m = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if m.get("type") != "state" or not m.get("counts"):
                continue
            left = sum(m["counts"])
            first = left if first is None else first
            if first - left >= 10:
                print(f"AI table is playing: {first} -> {left} cards in hands")
                return
    sys.exit("AI table did not play")

asyncio.run(watch())
print("smoke test ok")
