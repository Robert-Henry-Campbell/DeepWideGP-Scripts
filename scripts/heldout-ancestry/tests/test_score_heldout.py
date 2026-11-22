"""score_heldout.py with a fake scorer: shadow folders, the tuning-set check, records, refusals.

Run:  /home/robert/envs/widedeepgp/bin/python -m pytest tests/test_score_heldout.py -q
"""
import csv
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import score_heldout as S  # noqa: E402

FAKE = '''import argparse, csv, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--run"); ap.add_argument("--split", action="append"); ap.add_argument("--device")
ap.add_argument("--save_scores", action="store_true")
a = ap.parse_args()
truth = json.load(open(os.path.join(a.run, "truth.json")))
if truth.get("fail"):
    sys.exit(3)
for s in a.split:
    with open(os.path.join(a.run, f"scores_{s}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["label", "metric", "value"]); w.writeheader()
        for lab, v in truth[s].items():
            w.writerow(dict(label=lab, metric="auroc", value=v))
        w.writerow(dict(label="MEAN", metric="auroc", value=0.5))
    if a.save_scores:
        open(os.path.join(a.run, f"saved_{s}"), "w").write("1")
'''


def write_scores(path, aurocs, extra=()):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["label", "metric", "value", "snapshot"])
        w.writeheader()
        for lab, v in aurocs.items():
            w.writerow(dict(label=lab, metric="auroc", value=v, snapshot="best"))
            w.writerow(dict(label=lab, metric="bce", value=0.3, snapshot="best"))
        for row in extra:
            w.writerow(row)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    runs, code = tmp_path / "runs", tmp_path / "code"
    os.makedirs(code / "Fake")
    (code / "Fake" / "fake_score.py").write_text(FAKE)
    (code / "COMMITS.txt").write_text("Fake abc123\n")
    monkeypatch.setattr(S, "RUNS", str(runs))
    monkeypatch.setattr(S, "OUT", str(runs / "heldout_ancestry"))
    monkeypatch.setitem(S.MODELS, "fake", dict(runs="fake/ladder_{rung}/*", cwd="Fake", cmd=["fake_score.py"],
                                               py=sys.executable))

    def run(name, rung="hm3plus_n13199", val=None, finished=True, **truth):
        d = runs / "fake" / f"ladder_{rung}" / name
        os.makedirs(d)
        (d / "best.pt").write_text("weights")
        (d / "test_access.log").write_text("x")
        if finished:
            write_scores(str(d / "scores_val.csv"), val or {"D1": 0.61})
        (d / "truth.json").write_text(json.dumps({"val": val or {"D1": 0.61}, "asian": {"D1": 0.55},
                                                  "black": {"D1": 0.57}, **truth}))
        return str(d)
    return dict(runs=runs, code=str(code), run=run)


def test_aurocs_are_the_models_rows_only(tmp_path):
    p = str(tmp_path / "s.csv")
    write_scores(p, {"A": 0.6, "B": 0.7}, [dict(label="A", metric="auroc", value=0.55, snapshot="covariates_only"),
                                            dict(label="MEAN", metric="auroc", value=0.65, snapshot="best")])
    assert S.aurocs(p) == {"A": 0.6, "B": 0.7}


def test_the_shadow_links_the_run_but_not_what_a_scorer_writes(world):
    run = world["run"]("D1_large")
    d = S.shadow(run, "fake")
    assert sorted(os.listdir(d)) == ["best.pt", "truth.json"]
    assert os.path.realpath(os.path.join(d, "best.pt")) == os.path.join(run, "best.pt")
    assert S.shadow(run, "fake") == d                       # again: nothing changes


def test_a_run_is_scored_once_with_its_record_and_the_run_untouched(world):
    run = world["run"]("D1_large")
    before = sorted(os.listdir(run))
    assert S.score_run("fake", run, world["code"], "cpu").startswith("scored")
    d = S.shadow(run, "fake")
    rec = json.load(open(os.path.join(d, "heldout.json")))
    assert rec["cohorts"] == ["asian", "black"] and rec["tuning_auroc_max_difference"] == 0 and rec["code"][0] == "Fake abc123"
    assert os.path.exists(os.path.join(d, "saved_asian")) and not os.path.exists(os.path.join(d, "saved_val"))
    assert sorted(os.listdir(run)) == before                 # the run's own folder is never written
    assert S.score_run("fake", run, world["code"], "cpu") == "skipped"


def test_other_tuning_aurocs_stop_before_the_groups(world):
    run = world["run"]("D1_large", val={"D1": 0.61})
    with open(os.path.join(run, "truth.json")) as f:
        t = json.load(f)
    t["val"] = {"D1": 0.615}                                 # the model scored again gives 0.005 more
    json.dump(t, open(os.path.join(run, "truth.json"), "w"))
    with pytest.raises(RuntimeError, match="other AUROCs"):
        S.score_run("fake", run, world["code"], "cpu")
    d = S.shadow(run, "fake")
    assert not os.path.exists(os.path.join(d, "heldout.json")) and not os.path.exists(os.path.join(d, "saved_asian"))


def test_a_differing_set_of_diseases_and_a_failing_scorer_are_refused(world):
    run = world["run"]("D1_large")
    t = json.load(open(os.path.join(run, "truth.json")))
    t["val"] = {"D1": 0.61, "D2": 0.6}
    json.dump(t, open(os.path.join(run, "truth.json"), "w"))
    with pytest.raises(RuntimeError, match="other AUROCs"):
        S.score_run("fake", run, world["code"], "cpu")
    bad = world["run"]("D2_large", fail=True)
    with pytest.raises(RuntimeError, match="the scorer failed"):
        S.score_run("fake", bad, world["code"], "cpu")


def test_only_finished_runs_are_scored_and_a_dry_run_writes_nothing(world, capsys):
    world["run"]("D1_large")
    world["run"]("D2_large", finished=False)
    world["run"]("D3_large", rung="hm3plus_n26287")
    assert [os.path.basename(r) for r in S.finished_runs("fake", S.RUNGS)] == ["D1_large", "D3_large"]
    S.main(["--model", "fake", "--code", world["code"], "--dry_run"])
    out = capsys.readouterr().out
    assert "2 finished runs" in out and out.count("to score") == 2
    assert not os.path.exists(os.path.join(S.OUT, "shadow", "fake", "fake", "ladder_hm3plus_n13199", "D1_large", "heldout.log"))


def test_a_self_checking_scorer_is_called_once(world, capsys):
    assert S.main(["--model", "classic", "--code", world["code"], "--dry_run"]) == 0
    assert "score_genome.py --all --device cuda" in capsys.readouterr().out


def test_a_run_matched_by_two_rungs_patterns_is_refused(world, monkeypatch):
    """The top rung's name (hm3plus) prefixes every other rung's: a pattern must not catch their runs too."""
    monkeypatch.setitem(S.MODELS, "loose", dict(S.MODELS["fake"], runs="fake/ladder_{rung}*/*"))
    world["run"]("D1_large", rung="hm3plus_n13199")
    with pytest.raises(SystemExit, match="two rungs"):
        S.finished_runs("loose", ["hm3plus_n13199", "hm3plus"])
