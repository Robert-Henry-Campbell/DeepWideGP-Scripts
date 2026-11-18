"""A run read on the natural-prevalence tuning set and on random 1:1 subsets of it (methods.md, "A 1:1 case-control
run"): the whole-split AUROC against the mean and spread over the subsets, with the split's N_eff. Aggregates only.

  python balanced_eval.py --scores <dataset>/scores/<experiment>/<run>/val_best.npz --label I10
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from make_balanced_split import balanced_rows


def balanced_aurocs(logit, y, draws, seed):
    """AUROC on ``draws`` random 1:1 subsets of the rows (every row of the rarer class, as many of the other)."""
    rng = np.random.default_rng(seed)
    return np.array([roc_auc_score(y[r], logit[r]) for r in (balanced_rows(y, rng) for _ in range(draws))])


def n_eff(n_cases, n_controls):
    """4 x cases x controls / N: the information in a case-control sample of this mix."""
    return 4.0 * n_cases * n_controls / (n_cases + n_controls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="per-person scores written by score_snapshots.py --save_scores")
    ap.add_argument("--label", required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    z = np.load(a.scores)
    j = list(z["labels"]).index(a.label)
    logit, y = z["logit_population"][:, j].astype(float), z["y"][:, j].astype(float)
    ok = ~np.isnan(y)
    logit, y = logit[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    b = balanced_aurocs(logit, y, a.draws, a.seed)
    print(f"{a.label}: {n1} cases, {n0} controls, N_eff {n_eff(n1, n0):.0f}")
    print(f"AUROC on the whole split {roc_auc_score(y, logit):.4f}")
    print(f"AUROC on {a.draws} random 1:1 subsets (seed {a.seed}): mean {b.mean():.4f}, SD {b.std(ddof=1):.4f}, "
          f"range {b.min():.4f} to {b.max():.4f}")


if __name__ == "__main__":
    main()
