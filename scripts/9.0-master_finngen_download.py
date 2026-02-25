"""
Docstring for scripts.9.0-master_finngen_download

goal: 
1. merges the master df with the finngen manifest data including case counts
2. downloads the finngen GWAS sumstats


to run:
python scripts/9.1-finngen_download.py \
    --manifest data/9-finngen_download/finngen_R12_manifest.tsv \
        --master data/8-integrate_finngen_UKB/2025_casecount/FG_UKB_final_cleaned_pheno_df.csv \
            --dest data/9-finngen_download
"""

