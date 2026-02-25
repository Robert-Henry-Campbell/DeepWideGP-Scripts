"""
Docstring for ARD-DeepWideGP.scripts.7.0-master_prepro_discovery_snps
inputs:
phecodes for MVP phenotype GWAS
outputs:
full sumstats for MVP phenotype GWAS


inputs:
1.  MVP_UKB_final_cleaned_pheno.csv
    - has GBD ARD ICD10 phecode UKBcase10k MVPgwas 
2. directory of the MVP phenotype GWAS full sumstats files and their metadata

outputs: 
- write the full sumstats for each row of the MVP_UKB_final_cleaned_pheno.csv to output directory
- 6-grab_phenotype_discovery_GWAS directory default


parts:
6.1 Extract the sumstats for desired phecodes from MVP phenotype GWAS full sumstats files and place in ancestry specific directories
- structure is data/6-grab_phenotype_discovery_GWAS/{ancestry}/{phecode}{.txt.gz or metadata.txt}


"""
import os
import pandas as pd

#read the final cleaned phenotype file
pheno = pd.read_csv("data/4-integrate_MVP_UKB/2025_casecount/MVP_UKB_final_cleaned_pheno.csv")

#directory of MVP phenotype GWAS full sumstats files
mvp_gwas_dir = "data/5-MVP_download/

#output dir
output_dir = "data/6-grab_phenotype_discovery_GWAS/"
# TODO:#initialize logging, and remember to log summaries of what gets done at each stepf rom here on out-- no furtehr instruction on this but do it. 

# TODO:#read all *table_of_contents.txt files in the mvp_gwas_dir to get sumstat file names containing each phecode
# TODO:# for each table of contents, read it into a dataframe (tab separated). 
# TODO:# extract whichever column has entries like "MVP_R4.1000G_AGR.GIA.PheCodes_*" that's the file name column
# TODO:# split this column on "." and turn each into a new column to extract the phecode part. this also extracts ancestry info
# TODO:# store all these dataframes in a list for searching later

# TODO:#read pheno['MVP_Trait'] to get the desired phecodes, structured like 'Phe_XXX_YY', and the ICD10 category it belongs to 'ICD10_category' column
# TODO:# we wil extract phecodes and group them by ICD10 category

# TODO:#for each unqiue value of 'ICD10_category' in pheno, gather up all the 'MVP_Trait' values occuring in pheno rows with that unique value
# TODO:# the for each value of 'MVP_Trait' (for that icd10_category) search the list of table of context dataframes to find row containting matching phecode sumstats files instide the big files. 
# TODO:# be careful to matcht he MVP_Trait value to the phecode column in the table of contents dataframes
# TODO:# there can  be many matches for any given phecode: tehre will be .gz files with sumsats and metadata.txt files too, all for differen ancestries
# TODO:# createa  dictionary where the each ICD10_category value is key, and the value is a list containing each corresponding 'MVP_Trait', and each element of that list is a df where the rows are the table of contents rows matching that MVP_Trait phecode
# TODO:# for each key, for each value, for each element of the list, for each row of each element, extract the file it refers to and send it to output_dir/{ancestry}/{ICD10_category} directory

# TODO:# save a summary log of what was done, including any phecodes that were not found in any table of contents files
# TODO:# create a Df for each ancestry with rows for each ICD10_category and columns for each phecode found
# TODO:# save these Dfs as csv files in the output_dir