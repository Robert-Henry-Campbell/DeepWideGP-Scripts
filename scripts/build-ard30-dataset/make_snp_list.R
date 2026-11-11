#!/usr/bin/env Rscript
# HapMap3+ SNPs that are present in the lab's quality-controlled UK Biobank genotype files.
#
# Matches the LDpred2 HapMap3+ map (Prive et al.; figshare file 37802721) to the .pvar files of
# the imputed PLINK2 set by chromosome, position and alleles (either orientation), and writes the
# list that genostore.convert --extract reads. Only variant-level data are touched.
#
#   Rscript scripts/build-ard30-dataset/make_snp_list.R [data_dir] [pgen_dir]
#
# Outputs under <data_dir>/reference/:
#   map_hm3_plus.rds             the map as downloaded (cached)
#   hm3plus_in_ukb_pgen.tsv      one row per matched SNP, in genotype-file order
#   hm3plus_in_ukb_pgen.json     counts and checksums
suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "/mnt/sdg/robert/deepwidegp/data"
pgen_dir <- if (length(args) >= 2) args[2] else "/mnt/sdh/upamanyu/UKB/Genomics_PGEN"
map_url <- "https://figshare.com/ndownloader/files/37802721"

ref_dir <- file.path(data_dir, "reference")
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
map_path <- file.path(ref_dir, "map_hm3_plus.rds")
if (!file.exists(map_path)) {
  tmp <- paste0(map_path, ".part")
  download.file(map_url, tmp, mode = "wb", quiet = TRUE)
  file.rename(tmp, map_path)
}
info <- as.data.table(unclass(readRDS(map_path)))
stopifnot(all(c("chr", "pos", "a0", "a1", "rsid", "af_UKBB") %in% names(info)))

per_chr <- lapply(1:22, function(ch) {
  pv <- fread(file.path(pgen_dir, sprintf("UKB_chr%d.pvar", ch)), select = 1:5,
              col.names = c("chr", "pos", "id", "ref", "alt"), showProgress = FALSE)
  pv[, pgen_index := .I - 1L]                       # 0-based variant index in the .pgen
  j <- merge(info[chr == ch], pv, by = c("chr", "pos"))
  j <- j[(a0 == ref & a1 == alt) | (a0 == alt & a1 == ref)]
  stopifnot(!anyDuplicated(j$pos), !anyDuplicated(j$pgen_index))
  setorder(j, pgen_index)
  list(snps = j, n_pgen = nrow(pv), n_map = nrow(info[chr == ch]))
})
snps <- rbindlist(lapply(per_chr, `[[`, "snps"))
maf <- pmin(info$af_UKBB, 1 - info$af_UKBB)
key <- function(d) paste(d$chr, d$pos)
present <- key(info) %in% key(snps)

out <- snps[, .(CHROM = chr, POS = pos, ID = id, REF = ref, ALT = alt, rsid, a0, a1,
                af_UKBB, ld, block_id, pos_hg38, pgen_index)]
out_path <- file.path(ref_dir, "hm3plus_in_ukb_pgen.tsv")
fwrite(out, out_path, sep = "\t")

counts <- list(
  map_url = map_url,
  map_md5 = unname(tools::md5sum(map_path)),
  pgen_dir = pgen_dir,
  n_map = nrow(info),
  n_present = nrow(out),
  n_rsid_agree = sum(out$rsid == out$ID),
  n_missing = sum(!present),
  n_missing_maf_below_0.01 = sum(!present & maf < 0.01),
  n_missing_maf_at_least_0.01 = sum(!present & maf >= 0.01),
  min_maf_present = min(maf[present]),
  per_chromosome = lapply(seq_along(per_chr), function(i)
    list(chr = i, pgen_variants = per_chr[[i]]$n_pgen, map_variants = per_chr[[i]]$n_map,
         present = nrow(per_chr[[i]]$snps))),
  tsv_md5 = unname(tools::md5sum(out_path)),
  created = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
)
writeLines(jsonlite::toJSON(counts, auto_unbox = TRUE, pretty = TRUE, digits = 8),
           file.path(ref_dir, "hm3plus_in_ukb_pgen.json"))
cat(sprintf("HapMap3+ map %d | present in the genotype files %d (%.2f%%) | longest chr %d with %d | wrote %s\n",
            nrow(info), nrow(out), 100 * nrow(out) / nrow(info),
            which.max(sapply(per_chr, function(x) nrow(x$snps))),
            max(sapply(per_chr, function(x) nrow(x$snps))), out_path))
