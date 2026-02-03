"""
Docstring for scripts.8.0-master-integrate-finngen_UKB

#part 1: Integrate Finngen and UKB data into the master dataset
bash scripts/8.1-integrate_finngen_UKB.sh

#part 2: merge the mapped icd10 codes back into the master dataset with ARD/GBD/UKB/ etc
python scripts/8.2-merge_finngen_to_ARD_GBD_UKB.py \
--base-ard data/3-identify-UKB-casecount-ARDs/extract_casecount_ukb2025/GBD_ARD_UKB_2025_ICD0_casecounts.csv \
--finngen-map data/8-integrate_finngen_UKB/UKB_ICD10_finngen.tsv \
--output-dir data/8-integrate_finngen_UKB \
--output-prefix FG_UKB \
--ukb-case-threshold 10000
"""
