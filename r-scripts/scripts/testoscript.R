library(bigsnpr)
cat("hello\n")

info <- readRDS(runonce::download_file(
  "https://figshare.com/ndownloader/files/37802721",
  dir = "tmp-data", fname = "map_hm3_plus.rds"))
str(info)