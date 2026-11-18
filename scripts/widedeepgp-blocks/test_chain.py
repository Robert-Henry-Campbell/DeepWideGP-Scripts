"""The chain's recipe flags.  python -m pytest scripts/widedeepgp-blocks/test_chain.py -q"""
import importlib.util
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent


def _chain():
    spec = importlib.util.spec_from_file_location("run_chain", HERE / "chains" / "run_chain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_cell_s_flags_carry_the_objective_and_the_recipe_explicitly():
    c = _chain()
    f = c.flags("transformer", "S3", "prive431", "dosage", "ridge", c.TOP)
    pairs = dict(zip(f[::2], f[1::2]))
    assert (pairs["--sampler"], pairs["--gamma"]) == ("shuffled", "0")                      # 18 November 2026
    assert (pairs["--lr"], pairs["--l2"], pairs["--dropout"], pairs["--batch_size"], pairs["--input_mask"],
            pairs["--lr_schedule"], pairs["--stop_patience"], pairs["--stop_burnin"]) == \
        ("1e-4", "1e-3", "0.5", "64", "0.4", "plateau", "16", "2000")                      # Sigurdsson et al. 2023
    assert f[-4:] == c.OBJECTIVE == ["--sampler", "shuffled", "--gamma", "0"] and f.count("--sampler") == 1
    assert pairs["--folds"] == f"{c.DATA}/ard30/{c.TOP}/blocks/folds.npz"                    # fold-routed warm start
    assert "--folds" not in c.flags("transformer", "S3", "prive431", "dosage", "none", c.TOP)  # from zero: one copy
    rot = c.flags("transformer", "S3", "prive431", "dosage", "ridge", c.TOP, objective=c.ROTATION)   # the ablation cell
    p = dict(zip(rot[::2], rot[1::2]))
    assert (p["--sampler"], p["--gamma"], p["--lr"], p["--run_dir"]) == ("rotation", "0.5", "1e-4", c.RUNS)
    assert len(rot) == len(f) and rot[:-4] == f[:-4]


def test_the_sizes_and_the_launcher():
    c = _chain()
    assert [c.SIZES[s][0] for s in ("S1", "S2", "S3", "S4", "S5")] == [16, 32, 64, 128, 256]
    assert c.DEFAULTS == {"table_kappa": 1e-3, "table_decay": 0.0} and c.INIT == "ridge"     # a warm start kept, not erased
    assert subprocess.run(["bash", "-n", str(HERE / "chains" / "run_all.sh")]).returncode == 0


def test_the_comparator_phases_skip_by_product_and_run_only_the_missing_steps(tmp_path, monkeypatch):
    import os
    import pytest
    c = _chain()
    monkeypatch.setattr(c, "DATA", str(tmp_path / "datasets"))
    monkeypatch.setattr(c, "SCRIPTS", str(tmp_path / "scripts"))
    monkeypatch.setattr(c, "OUT", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()
    d = tmp_path / "datasets" / "ard30" / c.TOP
    (d / "blocks").mkdir(parents=True)
    gpu, cpu = c.lin_steps(c.TOP, "gpu"), c.lin_steps(c.TOP, "cpu")
    assert [os.path.basename(cmd[1]) for _, cmd in gpu] == ["make_folds.py", "build_block_ridge.py", "make_warm_start.py"]
    assert [os.path.basename(cmd[1]) for _, cmd in cpu] == ["check_block_ridge.py", "fit_block_linear.py"]
    assert [os.path.basename(p) for p, _ in gpu] == ["folds.npz", "blocks.json", "init_ridge_d16.pt"]
    assert [os.path.basename(p) for p, _ in cpu] == ["check_block_ridge.md", "blocks_linear.csv"]
    assert all(cmd[0] == c.PY and "--splits" in cmd for _, cmd in gpu[:2]) and gpu[2][1][-2:] == ["--embed_dim", "16"]
    with pytest.raises(ValueError):
        c.lin_steps(c.TOP, "both")
    product_of = {os.path.basename(cmd[1]): prod for prod, cmd in gpu + cpu}
    ran, outcome = [], {"fit_block_linear.py": 3}                        # the twin fails with exit 3

    def fake_sh(cmd, log_name, gpu=None, cwd=None, env=None):
        step = os.path.basename(cmd[1])
        ran.append((step, log_name))
        if step not in outcome:
            pathlib.Path(product_of[step]).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(product_of[step]).write_text("x")
        return outcome.get(step, 0)
    monkeypatch.setattr(c, "sh", fake_sh)
    (d / "blocks" / "folds.npz").write_text("x")                           # the block ridge is built already, as at
    (d / "blocks.json").write_text("x")                                    # the relaunch of 18 November: only the warm start runs
    assert c.lin(c.TOP, "gpu") == 0 and ran == [("make_warm_start.py", f"lin_{c.TOP}_gpu")]
    assert c.lin(c.TOP, "gpu") == 0 and len(ran) == 1                     # every product present: nothing runs
    assert c.lin(c.TOP, "cpu") == 3 and [r[0] for r in ran[1:]] == ["check_block_ridge.py", "fit_block_linear.py"]
    chain_log = (tmp_path / "out" / "chain.log").read_text()
    assert "fit_block_linear.py exit 3, blocks_linear.csv missing; chain stopped" in chain_log
    outcome["fit_block_linear.py"] = 0                                     # exit 0 without the product is a failure too
    assert c.lin(c.TOP, "cpu") == 1 and ran[-1][0] == "fit_block_linear.py"
    del outcome["fit_block_linear.py"]
    assert c.lin(c.TOP, "cpu") == 0
    assert "fit_block_linear.py exit 0, blocks_linear.csv missing; chain stopped" in (tmp_path / "out" / "chain.log").read_text()


def test_a_single_run_resumes_only_when_its_checkpoint_exists(tmp_path, monkeypatch):
    c = _chain()
    monkeypatch.setattr(c, "OUT", str(tmp_path))
    cmd = [c.PY, "main_multilabel.py", "--exp_name", "t0_x", "--tune_trials", "0"]
    assert c.single(cmd, "t0_x") == cmd
    (tmp_path / "t0_x" / "checkpoints").mkdir(parents=True)
    (tmp_path / "t0_x" / "checkpoints" / "last.ckpt").write_text("x")
    assert c.single(cmd, "t0_x") == cmd + ["--resume"] and cmd[-1] == "0"           # the given list is not changed
