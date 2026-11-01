"""Small cross-platform process / resource helpers (no psutil dependency)."""
from __future__ import annotations

import os
import signal
import subprocess
import time

PID_FILE = "pid"


def pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int):
    """Forcefully end a process and all of its children."""
    if not pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def read_pid(run_dir: str) -> int | None:
    try:
        with open(os.path.join(run_dir, PID_FILE)) as f:
            return int(f.read().strip() or 0)
    except Exception:
        return None


def run_alive(run_dir: str) -> bool:
    pid = read_pid(run_dir)
    return bool(pid) and pid_alive(pid)


def write_pid(run_dir: str):
    with open(os.path.join(run_dir, PID_FILE), "w") as f:
        f.write(str(os.getpid()))


def remove_pid(run_dir: str):
    try:
        if read_pid(run_dir) == os.getpid():
            os.remove(os.path.join(run_dir, PID_FILE))
    except Exception:
        pass


def system_ram_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 8.0


def gpu_mem_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def resource_limits(actor_device: str) -> dict:
    """Upper bounds for self-play so that a run cannot exhaust GPU memory or RAM."""
    ram = system_ram_gb()
    cpu = os.cpu_count() or 2
    if actor_device == "cuda":
        vram = gpu_mem_gb()
        # each GPU self-play process holds a CUDA context (~0.6 GB VRAM, ~1.5 GB RAM)
        max_actors = max(1, min(int((vram - 2.0) / 0.8), int((ram - 6) / 1.6), 6))
        max_total_envs = 1536
    else:
        max_actors = max(1, min(cpu, int((ram - 4) / 0.6), 32))
        max_total_envs = 64 * max_actors
    return {"max_actors": max_actors, "max_total_envs": max_total_envs, "ram_gb": round(ram, 1)}


def wait_gone(pid: int, timeout: float) -> bool:
    t = time.time() + timeout
    while time.time() < t:
        if not pid_alive(pid):
            return True
        time.sleep(0.5)
    return not pid_alive(pid)
