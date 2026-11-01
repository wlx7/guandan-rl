"""Plot win-rate curves of a training run (metrics.jsonl) into a PNG.

usage: python scripts/plot_metrics.py guandan_data/runs/main/metrics.jsonl out.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

SURFACE = "#102A22"
INK = "#EAF0EA"
MUTED = "#86A093"
SERIES = [("rule", "对规则 AI", "#B08A30"), ("base", "对基准模型", "#2A9FD0"), ("random", "对随机出牌", "#D65A50")]


def main(src, out):
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    evals = [r for r in rows if r.get("type") == "eval"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, facecolor=SURFACE)
    for ax, field, title, ylim in [(axes[0], "win_rate", "每局胜率", (0, 1)), (axes[1], "score", "平均每局升级数", None)]:
        ax.set_facecolor(SURFACE)
        for key, label, color in SERIES:
            pts = [(e["frames"] / 1e6, e[key][field]) for e in evals if key in e]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, lw=2, label=label)
                ax.plot(xs[-1], ys[-1], "o", color=color, ms=6, mec=SURFACE, mew=2)
        ax.axhline(0.5 if field == "win_rate" else 0, color=MUTED, lw=1, ls="--")
        if ylim:
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("训练样本（百万）", color=MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(color="#E9CF92", alpha=0.10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        leg = ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
