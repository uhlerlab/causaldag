library(pcalg)

amat = read.table('~/Documents/causaldag/tests/dag1.txt')
cpdag = dag2cpdag(as.matrix(amat))
write.table(cpdag*1, '~/Documents/causaldag/tests/cpdag1.txt', row.names=FALSE, col.names=FALSE)