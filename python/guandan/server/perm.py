"""Who may manage what.

The host (房主) is the computer that runs the server, plus any device that logged in with the
host password (房主密码). Other devices (phones on the LAN) can join and watch tables, open their
own tables (unless the host turned that off) and manage only the tables they opened; the training
center is for the host unless the host allows guests.

This keeps family and friends on the same Wi-Fi from closing each other's tables by accident; it
is not meant as protection against a determined attacker on the network (plain HTTP).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

from ..paths import home

FILE = "settings.json"
DEFAULTS = {"guest_create": True, "guest_train": False, "pin_salt": None, "pin_hash": None, "admin_tokens": []}
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_fails: dict[str, list[float]] = {}  # wrong passwords per address (rate limit)


def load() -> dict:
    try:
        with open(os.path.join(home(), FILE), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    if not isinstance(d, dict):
        d = {}
    out = dict(DEFAULTS)
    out.update({k: v for k, v in d.items() if k in DEFAULTS})
    if not isinstance(out["admin_tokens"], list):
        out["admin_tokens"] = []
    return out


def save(d: dict):
    p = os.path.join(home(), FILE)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _h(s: str, salt: str = "") -> str:
    return hashlib.sha256((salt + s).encode("utf-8")).hexdigest()


def is_local(host: str | None, own_ips=()) -> bool:
    """The request comes from the computer that runs the server."""
    return bool(host) and (host in LOOPBACK or host in own_ips)


def token_ok(token: str | None) -> bool:
    if not token:
        return False
    h = _h(token)
    return any(hmac.compare_digest(h, t) for t in load()["admin_tokens"])


def is_admin(host: str | None, token: str | None, own_ips=()) -> bool:
    return is_local(host, own_ips) or token_ok(token)


def pin_set() -> bool:
    return bool(load()["pin_hash"])


def set_pin(pin: str | None):
    """Set (or clear with an empty pin) the host password; logs out every remote host device."""
    d = load()
    pin = (pin or "").strip()
    if pin:
        salt = secrets.token_hex(8)
        d["pin_salt"], d["pin_hash"] = salt, _h(pin, salt)
    else:
        d["pin_salt"] = d["pin_hash"] = None
    d["admin_tokens"] = []
    save(d)


def login(pin: str, host: str) -> str:
    """A new host token for this device, or ValueError."""
    now = time.time()
    recent = [t for t in _fails.get(host, []) if now - t < 300]
    _fails[host] = recent
    if len(recent) >= 5:
        raise ValueError("密码错误次数太多，请 5 分钟后再试")
    d = load()
    if not d["pin_hash"]:
        raise ValueError("房主还没有设置房主密码")
    if not hmac.compare_digest(_h(str(pin).strip(), d["pin_salt"] or ""), d["pin_hash"]):
        _fails[host].append(now)
        raise ValueError("房主密码不对")
    token = secrets.token_urlsafe(24)
    d["admin_tokens"] = (d["admin_tokens"] + [_h(token)])[-20:]
    save(d)
    return token


def logout(token: str | None):
    if not token:
        return
    d = load()
    h = _h(token)
    d["admin_tokens"] = [t for t in d["admin_tokens"] if t != h]
    save(d)


def update(guest_create=None, guest_train=None):
    d = load()
    if guest_create is not None:
        d["guest_create"] = bool(guest_create)
    if guest_train is not None:
        d["guest_train"] = bool(guest_train)
    save(d)
