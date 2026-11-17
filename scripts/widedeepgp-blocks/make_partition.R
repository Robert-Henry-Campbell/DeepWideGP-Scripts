#!/usr/bin/env Rscript
# Nested LD-block partitions inside Prive's blocks, with bigsnpr::snp_ldsplit on the PGI arm's LD matrices.
#   Rscript make_partition.R [store=DIR] [reference=TSV] [ld_dir=DIR] [out=DIR] [targets=2000,5000] [chroms=21,22] [cores=8]
# Per chromosome (resumable: skipped once its file exists): <out>/work/chr<c>.tsv with chromosome-local 0-based
# block ids per target, and chr<c>.json with per-block diagnostics. Once every chromosome of the store has one:
# <out>/ldsplit<t>/blocks.tsv (CHROM ID block; global 0-based ids in store order) and make_partition.json.
# Reasoning and parameters: methods.md, section "Partitions".
.libPaths(c("/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/r-scripts/renv/library/linux-ubuntu-jammy/R-4.4/x86_64-pc-linux-gnu",
            .libPaths()))
suppressMessages({library(bigsnpr); library(data.table)})

opt <- list(store = "/mnt/sdg/robert/deepwidegp/data/genostore/hm3plus",
            reference = "/mnt/sdg/robert/deepwidegp/data/reference/hm3plus_in_ukb_pgen.tsv",
            ld_dir = "/mnt/sdg/robert/deepwidegp/data/pgi/ld/bigsnpr",
            out = "/mnt/sdg/robert/deepwidegp/data/reference/partitions",
            targets = "2000,5000", chroms = "", cores = "8")
for (a in commandArgs(trailingOnly = TRUE)) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (!kv[1] %in% names(opt)) stop("unknown argument: ", a)
  opt[[kv[1]]] <- paste(kv[-1], collapse = "=")
}
targets <- as.integer(strsplit(opt$targets, ",")[[1]])
stopifnot(length(targets) >= 1, all(targets >= 1), all(diff(targets) > 0))
THR_R2 <- 0.02
work_tsv <- function(c) file.path(opt$out, "work", sprintf("chr%s.tsv", c))

# snp_ldsplit settings for K blocks of m variants: sizes between a quarter and three times the mean (so K is
# always feasible), every K up to 2K reported (max_cost = Inf) and no hard cap on cut correlations (max_r2 = 1):
# a finer-than-LD split must cut strong pairs; the cost puts the cuts at the weakest LD.
settings <- function(m, K) list(min_size = max(1L, m %/% (4L * K)), max_size = min(m, ceiling(3 * m / K)), max_K = 2L * K)

# Largest-remainder apportionment of n among units of the given sizes, each unit at least its floor.
apportion <- function(n, sizes, floors = 1L) {
  k <- rep_len(as.integer(floors), length(sizes))
  r <- n - sum(k)
  stopifnot(r >= 0)
  q <- r * sizes / sum(sizes)
  k <- k + as.integer(floor(q))
  left <- r - sum(floor(q))
  if (left > 0) { i <- order(-(q - floor(q)))[seq_len(left)]; k[i] <- k[i] + 1L }
  k
}

# Labels 1..k of one run of variants in store order: the K-way split of those present in the LD matrix (columns
# ld_idx; NA = absent), the lowest-cost candidate at K or else at the nearest K offered; an absent variant joins
# its nearest present neighbour by position.
split_run <- function(corr, ld_idx, pos, K) {
  pres <- which(!is.na(ld_idx))
  if (!length(pres)) return(list(lab = rep(1L, length(ld_idx)), n = 1L, cost = 0))
  sizes <- length(pres); n <- 1L; cost <- 0
  if (K > 1 && length(pres) > 1) {
    s <- settings(length(pres), K)
    res <- snp_ldsplit(corr[ld_idx[pres], ld_idx[pres]], thr_r2 = THR_R2, min_size = s$min_size,
                       max_size = s$max_size, max_K = s$max_K, max_r2 = 1, max_cost = Inf)
    if (!is.null(res)) {
      best <- res[order(abs(res$n_block - K), res$cost), ][1, ]
      sizes <- best$all_size[[1]]; n <- as.integer(best$n_block); cost <- best$cost
    }
  }
  lab <- integer(length(ld_idx))
  lab[pres] <- rep(seq_along(sizes), sizes)
  absent <- which(is.na(ld_idx))
  if (length(absent)) lab[absent] <- lab[pres][vapply(pos[absent], function(p) which.min(abs(pos[pres] - p)), 1L)]
  list(lab = lab, n = n, cost = cost)
}

ref <- fread(opt$reference, select = c("ID", "block_id"))
prive <- ref[, .N, by = block_id]                       # Prive block sizes over the reference's variants
# Blocks per Prive block and level: proportional to SNP count, each at least one and at least its previous level's
# count, so the levels nest and the genome-wide totals are the targets.
alloc <- Reduce(function(prev, t) apportion(t, prive$N, prev), targets, accumulate = TRUE, init = 1L)[-1]

do_chrom <- function(c) {
  t0 <- Sys.time()
  v <- fread(file.path(opt$store, sprintf("chr%s.variants.tsv", c)), select = c("#CHROM", "POS", "ID"))
  setnames(v, "#CHROM", "CHROM")
  block <- ref$block_id[match(v$ID, ref$ID)]
  if (anyNA(block)) stop(sprintf("chr%s: %d store variants are not in the reference table", c, sum(is.na(block))))
  if (anyDuplicated(rle(block)$values)) stop(sprintf("chr%s: Prive blocks are not contiguous in store order", c))
  snp <- fread(file.path(opt$ld_dir, sprintf("chr%s.snpinfo.tsv", c)), select = "SNP")$SNP
  corr <- readRDS(file.path(opt$ld_dir, sprintf("chr%s.corr.rds", c)))
  if (ncol(corr) != length(snp)) stop(sprintf("chr%s: LD matrix has %d columns, snpinfo %d rows", c, ncol(corr), length(snp)))
  ld_idx <- match(v$ID, snp)
  labs <- matrix(0L, nrow(v), length(targets)); base <- integer(length(targets)); diags <- list()
  for (b in unique(block)) {
    rows <- which(block == b)
    kb <- vapply(alloc, `[`, 1L, match(b, prive$block_id))
    parent <- rep(1L, length(rows))
    for (l in seq_along(targets)) {
      units <- split(seq_along(rows), parent)              # the previous level's blocks, in order
      ks <- apportion(max(kb[l], length(units)), lengths(units), 1L)
      lab <- integer(length(rows)); off <- 0L; cost <- 0
      for (u in seq_along(units)) {
        s <- split_run(corr, ld_idx[rows[units[[u]]]], v$POS[rows[units[[u]]]], ks[u])
        lab[units[[u]]] <- off + s$lab; off <- off + s$n; cost <- cost + s$cost
      }
      diags[[length(diags) + 1]] <- data.table(block_id = b, target = targets[l], m = length(rows), k_target = kb[l], k = off, cost = cost)
      labs[rows, l] <- base[l] + lab; base[l] <- base[l] + off; parent <- lab
    }
  }
  dir.create(file.path(opt$out, "work"), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(list(chrom = c, n_variants = nrow(v), n_absent_from_ld = sum(is.na(ld_idx)), blocks = rbindlist(diags)),
                       sub("tsv$", "json", work_tsv(c)), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  out <- data.table(CHROM = v$CHROM, ID = v$ID)
  for (l in seq_along(targets)) out[[sprintf("ldsplit%d", targets[l])]] <- labs[, l] - 1L
  fwrite(out, paste0(work_tsv(c), ".tmp"), sep = "\t")
  file.rename(paste0(work_tsv(c), ".tmp"), work_tsv(c))   # the file's presence marks the chromosome done
  message(sprintf("chr%s: %d variants, %d absent from the LD matrix, blocks %s, %.0f s", c, nrow(v), sum(is.na(ld_idx)),
                  paste(base, collapse = "/"), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  TRUE
}

assemble <- function(cs) {
  w <- rbindlist(lapply(cs, function(c) fread(work_tsv(c), colClasses = list(character = c("CHROM", "ID")))))
  info <- lapply(cs, function(c) jsonlite::read_json(sub("tsv$", "json", work_tsv(c)), simplifyVector = TRUE))
  blocks <- rbindlist(lapply(info, `[[`, "blocks"))
  absent <- setNames(lapply(info, `[[`, "n_absent_from_ld"), cs)
  for (t in targets) {
    col <- sprintf("ldsplit%d", t)
    nb <- w[, max(get(col)) + 1L, by = CHROM]$V1                  # blocks per chromosome, in store order
    w[, (col) := get(col) + rep(cumsum(c(0L, nb[-length(nb)])), w[, .N, by = CHROM]$N)]
    sizes <- tabulate(w[[col]] + 1L)
    bt <- blocks[target == t]
    dir.create(file.path(opt$out, col), recursive = TRUE, showWarnings = FALSE)
    fwrite(w[, .(CHROM, ID, block = get(col))], file.path(opt$out, col, "blocks.tsv"), sep = "\t")
    jsonlite::write_json(list(
      partition = col, target = t, n_blocks = length(sizes), n_variants = nrow(w),
      blocks_per_chromosome = setNames(as.list(nb), cs), n_absent_from_ld = list(total = sum(unlist(absent)), per_chromosome = absent),
      block_size = as.list(quantile(sizes, c(0, .05, .25, .5, .75, .95, 1), names = TRUE)),
      prive_blocks_off_target = sum(bt$k != bt$k_target), total_cost = sum(bt$cost),
      parameters = list(thr_r2 = THR_R2, max_r2 = 1, max_cost = "Inf", min_size = "max(1, floor(m / (4 K)))",
                        max_size = "min(m, ceil(3 m / K))", max_K = "2 K", candidate = "lowest cost at K, else nearest K",
                        allocation = "largest remainder, proportional to SNP count, each block >= 1 and >= its count at the level above"),
      inputs = list(store = opt$store, reference = opt$reference, ld_dir = opt$ld_dir, targets = targets,
                    bigsnpr = as.character(packageVersion("bigsnpr")), R = R.version.string),
      prive_blocks = bt[, .(block_id, m, k_target, k, cost)]),
      file.path(opt$out, col, "make_partition.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
    message(sprintf("%s: %d blocks, sizes %s", col, length(sizes), paste(quantile(sizes, c(0, .5, 1)), collapse = "/")))
  }
}

cs <- sub("^chr(.+)\\.variants\\.tsv$", "\\1", list.files(opt$store, "^chr.+\\.variants\\.tsv$"))
cs <- cs[order(suppressWarnings(as.numeric(cs)), cs)]
todo <- if (nzchar(opt$chroms)) strsplit(opt$chroms, ",")[[1]] else cs
stopifnot(all(todo %in% cs))
todo <- todo[!file.exists(work_tsv(todo))]
if (length(todo)) {
  res <- parallel::mclapply(todo, function(c) tryCatch(do_chrom(c), error = function(e) conditionMessage(e)),
                            mc.cores = min(as.integer(opt$cores), length(todo)), mc.preschedule = FALSE)
  failed <- unlist(Filter(is.character, res))
  if (length(failed)) stop(paste(failed, collapse = "\n"))
}
if (all(file.exists(work_tsv(cs)))) assemble(cs) else message(sprintf("%d of %d chromosomes done; no assembly yet",
                                                                     sum(file.exists(work_tsv(cs))), length(cs)))
