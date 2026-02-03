
#!/usr/bin/env bash
set -euo pipefail

# Goal: select FinnGen endpoints and add them to the master dataframe of ARDs using the phenotype mapping tool.

#OLD strategy:
#1. merge the top ARD df with the finngen ICD10-finnendpoint. 
#2. write the output to data/8-integrate_finngen_UKB
#3. write logs, following strategy of 4-integrate...
#4. write tests


ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Inputs
ARD_ICD10_CSV="${ROOT_DIR}/data/3-identify-UKB-casecount-ARDs/bothsex_ARD_GBD_ICD10_exploded_20250911_201039.csv"
FG_SOURCE_TSV="${ROOT_DIR}/external/upstream/data/FINNGEN_ENDPOINTS_DF12_Final_2023-05-17.tsv"
MAP_SOURCE_TSV="${ROOT_DIR}/external/upstream/data/UKB_PHENOME_ICD10_PHECODE_MAP_20200109_and_Phecode_map_v1_2_icd10_beta.combined.tsv"
PHENOMATCH_PY="${ROOT_DIR}/external/upstream/Scripts/phenomatch.py"

# Outputs
OUT_DIR="${ROOT_DIR}/data/8-integrate_finngen_UKB"
OUT_TSV="${OUT_DIR}/UKB_ICD10_finngen.tsv"
PHENOMATCH_INPUT_TSV="${OUT_DIR}/bothsex_ARD_GBD_ICD10_explo__phenomatch_input.tsv"

mkdir -p "${OUT_DIR}"

for f in "${ARD_ICD10_CSV}" "${FG_SOURCE_TSV}" "${MAP_SOURCE_TSV}" "${PHENOMATCH_PY}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing required file: ${f}" >&2
    exit 1
  fi
done

python3 -c "import pandas, numpy, rich" >/dev/null 2>&1 || {
  echo "ERROR: missing Python deps (pandas/numpy/rich)." >&2
  echo "Install with: pip3 install -r ${ROOT_DIR}/requirements.txt" >&2
  exit 1
}

# phenomatch.py requires a phenotype column AND a phenotype-type column.
# Build a small TSV from the ARD CSV: ICD10_explo + constant data_type=icd10.
python3 - "${ARD_ICD10_CSV}" "${PHENOMATCH_INPUT_TSV}" <<'PY'
import csv
import sys

in_path, out_path = sys.argv[1], sys.argv[2]

with open(in_path, newline="", encoding="utf-8") as f, open(
    out_path, "w", newline="", encoding="utf-8"
) as out:
    reader = csv.DictReader(f)
    if "ICD10_explo" not in (reader.fieldnames or []):
        raise SystemExit("ERROR: input CSV missing required column: ICD10_explo")

    writer = csv.writer(out, delimiter="\t")
    writer.writerow(["ICD10_explo", "data_type"])

    seen = set()
    for row in reader:
        icd = (row.get("ICD10_explo") or "").strip()
        if not icd or icd in seen:
            continue
        seen.add(icd)
        writer.writerow([icd, "icd10"])
PY

python3 "${PHENOMATCH_PY}" \
  --main-table phecode \
  --phecode-source "${PHENOMATCH_INPUT_TSV}" \
  --fg-source "${FG_SOURCE_TSV}" \
  --map-source "${MAP_SOURCE_TSV}" \
  --pheno-pheno-col ICD10_explo \
  --pheno-type-col data_type \
  --fg-pheno-col NAME \
  --map-pheno-col phecode \
  --fg-inc-col INCLUDE \
  --map-icd-col ICD10 \
  --fg-icd-col HD_ICD_10 \
  --out "${OUT_TSV}"

echo "Wrote: ${OUT_TSV}"
