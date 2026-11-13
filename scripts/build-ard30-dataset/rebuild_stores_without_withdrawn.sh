#!/bin/bash
# Rebuild the two genotype stores without the participants on UK Biobank's withdrawal list
# (25 April 2024): 120 in genostore/hm3plus and 16 in genostore/hm3plus_asian_black. Robert
# asked for this on 12 Nov 2026. No sample table contains them, so no result changes.
#
# For each store: keep list (.psam participants minus the withdrawal list) -> genostore.convert
# into <store>.rebuild -> genostore.convert verify -> check_rebuilt_store.py against the old
# store (participants, tables, variants, genotypes of 2,000 participants on chromosomes 1, 6
# and 22). Only when every step passes and no process of this user has the old store open is the
# old store deleted and the new one moved into its place, so every dataset's store.json still
# points at it. Resumable: rerun after an interruption.
set -u -o pipefail
umask 077
D=/mnt/sdg/robert/deepwidegp/data/genostore
DS=/mnt/sdg/robert/deepwidegp/data/datasets/ard30
R=/mnt/sdg/robert/deepwidegp/DeepWideGP
HERE=/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/scripts/build-ard30-dataset
GS=/mnt/sdg/robert/miniforge3/envs/gsem-py/bin/python      # has pgenlib
PY=/home/robert/envs/widedeepgp/bin/python
PGEN=/mnt/sdh/upamanyu/UKB/Genomics_PGEN
WITHDRAWN=/mnt/sdh/upamanyu/UKB/Proteomics_Olink/data/remove_eids_15181_25_Apr_2024.csv
EXTRACT=/mnt/sdg/robert/deepwidegp/data/reference/hm3plus_in_ukb_pgen.tsv
LOG=$D/rebuild_without_withdrawn.log

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

rebuild() {   # store name, pgen dir, sample tables...
    local name=$1 pgen=$2
    shift 2
    local old=$D/$name new=$D/$name.rebuild keep=$D/keep_${name}_not_withdrawn.txt
    if [ ! -e "$old/meta.json" ]; then log "$name: no store at $old, nothing to do"; return 1; fi
    if grep -q "not_withdrawn" "$old/meta.json"; then log "$name: already rebuilt"; return 0; fi
    log "$name: keep list"
    (cd "$R" && "$GS" - "$pgen/UKB_chr1.psam" "$WITHDRAWN" "$keep" <<'EOF'
import sys
from genostore.convert import read_psam_ids
psam, withdrawn, out = sys.argv[1:]
gone = {l.strip() for l in open(withdrawn) if l.strip().isdigit()}
ids = read_psam_ids(psam)
keep = [s for s in ids if s not in gone]
with open(out, "w") as f:
    f.write("".join(s + "\n" for s in keep))
print(f"{len(ids)} participants in the .psam, {len(ids) - len(keep)} on the withdrawal list, {len(keep)} kept")
EOF
    ) 2>&1 | tee -a "$LOG" || return 1
    log "$name: converting into $new"
    (cd "$R" && "$GS" -m genostore.convert --pgen-dir "$pgen" --pattern "UKB_chr{chrom}" --chroms $(seq 1 22) \
        --extract "$EXTRACT" --keep-file "$keep" --out "$new") > "$D/${name}_rebuild_convert.log" 2>&1
    local rc=$?
    log "$name: convert exit $rc (details in ${name}_rebuild_convert.log)"
    [ "$rc" -eq 0 ] || return 1
    (cd "$R" && "$GS" -m genostore.convert verify --out "$new") > "$D/${name}_rebuild_verify.log" 2>&1
    rc=$?
    log "$name: verify exit $rc; $(grep -c ' -> 0 mismatches' "$D/${name}_rebuild_verify.log") of 22 chromosomes with 0 mismatches"
    [ "$rc" -eq 0 ] && [ "$(grep -c ' -> 0 mismatches' "$D/${name}_rebuild_verify.log")" -eq 22 ] || return 1
    log "$name: checking against the old store"
    CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/check_rebuilt_store.py" --old "$old" --new "$new" --withdrawn "$WITHDRAWN" \
        --tables "$@" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: checks failed; both stores left in place"; return 1; }
    for attempt in $(seq 1 30); do
        if ! ls -l /proc/[0-9]*/fd 2>/dev/null | grep -qF "$old/"; then break; fi
        log "$name: the old store is open in another process; waiting"
        sleep 60
    done
    if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -qF "$old/"; then log "$name: still open; not swapped"; return 1; fi
    mv "$old" "$old.with_withdrawn" && mv "$new" "$old" && rm -rf "${D:?}/${name:?}.with_withdrawn"
    log "$name: rebuilt store in place; the store with withdrawn participants is deleted"
}

log "start"
rebuild hm3plus_asian_black "$PGEN/Asian_Black" "$DS"/hm3plus_asian_black/samples_*.tsv || { log "stopped"; exit 1; }
rebuild hm3plus "$PGEN" "$DS"/hm3plus/samples_*.tsv "$DS"/hm3plus_proxy8*/samples_*.tsv || { log "stopped"; exit 1; }
log "done"
