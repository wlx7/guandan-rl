import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client():
    home = tempfile.mkdtemp()
    os.environ["GUANDAN_HOME"] = home
    from guandan.model import GuandanNet, PRESETS, save_checkpoint
    save_checkpoint(os.path.join(home, "models", "test.pt"), GuandanNet(PRESETS["tiny"]), half=True)
    from fastapi.testclient import TestClient
    from guandan.server.app import app
    with TestClient(app) as c:
        yield c


def test_info_and_models(client):
    info = client.get("/api/info").json()
    assert info["recommended_preset"] == "medium"
    models = client.get("/api/models").json()
    assert any(m["id"].endswith("test.pt") for m in models)


@pytest.mark.parametrize("length", ["quick", "full"])
def test_teach_round_via_websocket(client, length):
    r = client.post("/api/room/create", json={"mode": "teach", "length": length, "model": "models/test.pt", "ai_speed": 3}).json()
    token = r["token"]
    with client.websocket_connect(f"/ws?token={token}") as ws:
        played = 0
        for _ in range(3000):
            m = ws.receive_json()
            if m["type"] == "state":
                if m["phase"] >= 2:
                    assert m["result"]["gain"] in (1, 2, 3)
                    assert set(m["hands"]) == {"1", "2", "3"}  # teaching mode can see everybody
                    return
                if m["current"] == m["you"] and m["status"] == "playing" and m["phase"] in (0, 1):
                    ws.send_json({"action": "hint"})
            elif m["type"] == "hints":
                best = m["hints"][0]
                if best["kind"] == 11:
                    ws.send_json({"action": "return", "card": best["cards"][0]})
                elif best["kind"] == 0:
                    ws.send_json({"action": "pass"})
                else:
                    ws.send_json({"action": "play", "cards": best["cards"], "kind": best["kind"], "key": best["key"]})
                played += 1
            elif m["type"] == "error":
                # stale hint after state changed; ask again
                pass
        raise AssertionError("round did not finish")


def test_real_mode_hides_hands(client):
    r = client.post("/api/room/create", json={"mode": "real", "length": "quick", "model": "models/test.pt",
                                              "seats": [{"kind": "human"}, {"kind": "ai"}, {"kind": "ai"}, {"kind": "ai"}]}).json()
    with client.websocket_connect(f"/ws?token={r['token']}") as ws:
        for _ in range(20):
            m = ws.receive_json()
            if m["type"] == "state" and m["status"] == "playing":
                assert m["hands"] == {}
                assert m["hint_available"] is False
                ws.send_json({"action": "hint"})
                break
        for _ in range(50):
            m = ws.receive_json()
            if m["type"] == "error":
                assert "实战" in m["message"] or "轮到" in m["message"]
                return


def _state(ws, cond, n=200):
    for _ in range(n):
        m = ws.receive_json()
        if m.get("type") == "state" and cond(m):
            return m
    raise AssertionError("state not reached")


def test_several_tables_pause_and_close(client):
    for r in client.get("/api/rooms").json()["rooms"]:
        client.post("/api/room/close", json={"room": r["id"]})
    a = client.post("/api/room/create", json={"mode": "teach", "length": "quick", "model": "models/test.pt", "ai_speed": 5}).json()
    b = client.post("/api/room/create", json={"mode": "real", "length": "quick", "model": "models/test.pt", "ai_speed": 5,
                                              "seats": [{"kind": "ai"}] * 4}).json()
    listed = {r["id"]: r for r in client.get("/api/rooms").json()["rooms"]}
    assert set(listed) == {a["room"], b["room"]}
    assert listed[a["room"]]["humans"] == 1 and listed[b["room"]]["humans"] == 0
    # the player leaves table a: it pauses (a spectator sees it) and does not move on
    with client.websocket_connect(f"/ws?token={a['token']}") as ws:
        _state(ws, lambda m: m["status"] == "playing")
    with client.websocket_connect(f"/ws?room={a['room']}") as spec:
        m = _state(spec, lambda m: m["paused"])
        assert m["you"] == -1
    # pause by hand from the list, then resume
    client.post("/api/room/pause", json={"room": b["room"], "paused": True})
    assert {r["id"]: r for r in client.get("/api/rooms").json()["rooms"]}[b["room"]]["pause_by"] == "房主"
    client.post("/api/room/pause", json={"room": b["room"], "paused": False})
    # closing one table leaves the other
    client.post("/api/room/close", json={"room": a["room"]})
    assert [r["id"] for r in client.get("/api/rooms").json()["rooms"]] == [b["room"]]
    with client.websocket_connect(f"/ws?token={a['token']}") as ws:
        assert ws.receive_json()["fatal"]
    client.post("/api/room/close", json={"room": b["room"]})


def test_player_ends_the_table(client):
    r = client.post("/api/room/create", json={"mode": "teach", "length": "quick", "model": "models/test.pt", "ai_speed": 5}).json()
    with client.websocket_connect(f"/ws?token={r['token']}") as ws:
        _state(ws, lambda m: m["status"] == "playing")
        ws.send_json({"action": "end"})
        for _ in range(50):
            m = ws.receive_json()
            if m.get("type") == "closed":
                break
        assert m["type"] == "closed" and "结束" in m["message"]
    assert r["room"] not in [x["id"] for x in client.get("/api/rooms").json()["rooms"]]
