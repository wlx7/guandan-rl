"""Run the NJUPT rule bot of the 1st national GuanDan AI competition in the version with the
crash fixes collected in github.com/dashidhy/DanLM (baselines/fin-njupt-guandan-ai), which
DanLM found to be the strongest competition bot. The glue is the bot's own
lxw/rule_client.py, using the fixed top-level action.py.

    python njupt_launch.py --src <DanLM/baselines/fin-njupt-guandan-ai> --name nj1
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", default="njupt")
    args = ap.parse_args()
    os.chdir(os.path.abspath(args.src))  # the bot writes its log next to itself
    sys.path.insert(0, os.path.abspath(args.src))
    from ws4py.client.threadedclient import WebSocketClient  # noqa: PLC0415

    from action import Action  # noqa: PLC0415
    from state import State  # noqa: PLC0415

    class Client(WebSocketClient):
        def __init__(self, url):
            super().__init__(url)
            self.state = State()
            self.action = Action(args.name)
            self.position = None

        def closed(self, code, reason=None):
            os._exit(0)

        def received_message(self, message):
            message = json.loads(str(message))
            self.state.parse(message)
            self.position = self.state._myPos
            if "actionList" in message:
                rest = [message["publicInfo"][i]["rest"] for i in range(4)]
                try:
                    idx = self.action.parse(message, self.position, rest)
                except Exception as e:  # noqa: BLE001
                    print("njupt error:", e, file=sys.stderr, flush=True)
                    idx = 0
                self.send(json.dumps({"actIndex": int(idx)}))
            if message.get("stage") == "gameResult":
                self.close()

    ws = Client(f"ws://127.0.0.1:23456/game/{args.name}")
    ws.connect()
    ws.run_forever()


if __name__ == "__main__":
    main()
