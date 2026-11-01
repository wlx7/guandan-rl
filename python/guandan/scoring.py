"""Optional score / multiplier system ("计分倍数").

Round score = base x multiplier for the winning team (negative for the losers):
* base: 3 for 双上, 2 when the partner is 3rd, 1 when the partner is last (same as the level gain)
* multiplier: starts at 1 and doubles for each bomb of the doubling categories, up to a cap.
  Categories: [4-5 card bomb, 6+ card bomb, straight flush, joker bomb].
"""

PRESETS = {
    "none": None,
    "standard": {"doubles": [False, True, True, True], "cap": 16,
                 "label": "标准倍数", "desc": "6 张及以上炸弹、同花顺、天王炸每出一个翻一倍，封顶 16 倍"},
    "all": {"doubles": [True, True, True, True], "cap": 64,
            "label": "全炸翻倍", "desc": "每出一个炸弹（含同花顺、天王炸）翻一倍，封顶 64 倍"},
}


def get(name):
    return PRESETS.get(name or "none")
