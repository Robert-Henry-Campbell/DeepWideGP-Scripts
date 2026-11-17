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
    rot = c.flags("transformer", "S3", "prive431", "dosage", "ridge", c.TOP, objective=c.ROTATION)   # the ablation cell
    p = dict(zip(rot[::2], rot[1::2]))
    assert (p["--sampler"], p["--gamma"], p["--lr"], p["--run_dir"]) == ("rotation", "0.5", "1e-4", c.RUNS)
    assert len(rot) == len(f) and rot[:-4] == f[:-4]


def test_the_sizes_and_the_launcher():
    c = _chain()
    assert [c.SIZES[s][0] for s in ("S1", "S2", "S3", "S4", "S5")] == [16, 32, 64, 128, 256]
    assert subprocess.run(["bash", "-n", str(HERE / "chains" / "run_all.sh")]).returncode == 0
