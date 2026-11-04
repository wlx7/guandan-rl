import json
import os
import shutil


def test_model_list_labels_hiding_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANDAN_HOME", str(tmp_path))
    from guandan import paths
    builtin = os.path.join(paths.builtin_models_dir(), "guandan-medium.pt")
    for run, files in {"a": ["model.pt", "model_100k.pt"], "b": ["model.pt"]}.items():
        d = tmp_path / "runs" / run
        d.mkdir(parents=True)
        for f in files:
            shutil.copyfile(builtin, d / f)
    (tmp_path / "models").mkdir()
    shutil.copyfile(builtin, tmp_path / "models" / "v1.pt")

    ids = [m["id"] for m in paths.list_models()]
    assert "runs/a/model.pt" in ids and "runs/b/model.pt" in ids and "models/v1.pt" in ids
    assert "runs/a/model_100k.pt" not in ids  # snapshots only with all=True
    assert "runs/a/model_100k.pt" in [m["id"] for m in paths.list_models(all=True)]

    paths.set_model_entry("runs/a", label="最强", hidden=None)
    paths.set_model_entry("runs/b", hidden=True)
    paths.set_model_entry("models/v1.pt", label="第一版")
    ms = {m["id"]: m for m in paths.list_models()}
    assert ms["runs/a/model.pt"]["label"] == "最强"
    assert ms["models/v1.pt"]["label"] == "第一版"
    assert "runs/b/model.pt" not in ms
    assert paths.resolve_model("runs/b/model.pt")  # hidden models still resolve
    assert ms["builtin/guandan-medium.pt"]["label"].startswith("内置模型 v")

    paths.set_model_entry("runs/b", hidden=False)
    paths.set_model_entry("runs/a", label="")
    ms = {m["id"]: m for m in paths.list_models()}
    assert "runs/b/model.pt" in ms and ms["runs/a/model.pt"]["label"] == "训练 a"
    d = json.loads((tmp_path / "model_list.json").read_text(encoding="utf-8"))
    assert d["hidden"] == [] and "runs/a" not in d["labels"]
