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
