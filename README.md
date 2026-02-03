
Script explanations:

################
5.1-MVP_phecode_GWAS_download.py
- downloads all the MVP phecode GWAS sumstats to directory of choice. 
- to use: 
python scripts/5.1-MVP_phecode_GWAS_download.py --base-url "https://ftp.ncbi.nlm.nih.gov/dbgap/studies/phs002453/analyses/GIA/" --dest "/mnt/sdg/robert/deepwidegp/ARD-DeepWideGP/data/5-MVP_download"
################

Testing (pytest):
- Install: `python -m pip install -r requirements.txt -r requirements-dev.txt`
- Run: `pytest`
