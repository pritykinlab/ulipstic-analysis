library(qvalue)
library(ggplot2)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)

if (length(args) != 1) {
   stop("wrong number of arguments")
}

corr_df = read.csv(args[1])
pvals <- corr_df[["pval"]]
qval_obj <- qvalue(p = pvals)
corr_df$qval <- qval_obj$qvalues    
corr_df$qval_sig <- corr_df["qval"] <= 0.05
print(sum(corr_df$qval_sig))
write.csv(corr_df %>% subset(select = -X, qval_sig),
          args[1], quote=FALSE, row.names = FALSE)
