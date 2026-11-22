#!/bin/bash
# The committed code the held-out scoring runs from (CLAUDE.md 8): branch widening of every repo, side by side as the
# chains lay them out (GLN is the EIR fork), with COMMITS.txt. One folder per set of commits; prints its path.
set -eu
ROOT=/mnt/sdg/robert/deepwidegp
OUT=$ROOT/runs/heldout_ancestry
declare -A REPO=([DeepWideGP]=$ROOT/DeepWideGP [KelemenMLP]=$ROOT/KelemenMLP [Delphi]=$ROOT/Delphi
                 [GLN]=$ROOT/GLN/EIR [PRSformer]=$ROOT/PRSformer [scPRS]=$ROOT/scPRS)
NAMES="DeepWideGP KelemenMLP Delphi GLN PRSformer scPRS"
commits=$(for n in $NAMES; do echo "$n $(git -C "${REPO[$n]}" rev-parse --short widening)"; done)
CODE=$OUT/code_$(echo "$commits" | sha256sum | cut -c1-10)
if [ ! -d "$CODE" ]; then
  rm -rf "${CODE:?}.tmp"
  for n in $NAMES; do
    mkdir -p "$CODE.tmp/$n"
    git -C "${REPO[$n]}" archive widening | tar -x -C "$CODE.tmp/$n"
  done
  echo "$commits" > "$CODE.tmp/COMMITS.txt"
  mv "$CODE.tmp" "$CODE"
fi
echo "$CODE"
