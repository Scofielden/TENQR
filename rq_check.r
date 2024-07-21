# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

# set the working directory
datapath <- 'D:/multilayer/外汇/Code_new/Data'
setwd(datapath)

# install and load packages
libraries  = c("quantreg", 'R.matlab')
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# load data
YpartV = readMat("YpartV.mat"); Xall = readMat("Xall.mat")
pv = readMat("pv.mat")
Xall = Xall$Xall
pv=pv$pv



all = matrix(0, ncol = 2*length(pv), nrow = dim(Xall)[2])     
all.resid = matrix(0, ncol = length(pv), nrow = dim(Xall)[1])  

# resultnamec <- c('coef', 'stdErr', 't-value', 'p-value')
regresult_pos <- matrix(0, length(pv), 4);
regresult_neg <- matrix(0, length(pv), 4); 

for(i in c(1:length(pv)))
{
  qr.sifi = rq(YpartV$YpartV ~ Xall[,-1], pv[i])  
  Z = summary(qr.sifi);
  all[, (i-1)*2+ c(1:2)] = Z$coefficients[, 1:2]  
  all.resid[, i] = Z$residuals;                   
  regresult_pos[i,1:4] <- Z$coefficients[3,1:4]
  regresult_neg[i,1:4] <- Z$coefficients[4,1:4]  
}


write.table(all, file           = 'rq_result.txt'    , col.names = F, row.names = F, sep ="\t")
write.table(all.resid, file     = 'rq_resid.txt'     , col.names = F, row.names = F, sep ="\t")
write.table(regresult_pos, file = 'regresult_pos.txt', row.names = F, col.names = F, sep ="\t")
write.table(regresult_neg, file = 'regresult_neg.txt', row.names = F, col.names = F, sep ="\t")







