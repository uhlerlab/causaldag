library(pcalg)

amat = read.table('~/Documents/causaldag/tests/dag1.txt')
intervention_mat = as.matrix(read.table('~/Documents/causaldag/tests/interventions.txt'))
interventions = list()
for (i in 1:nrow(intervention_mat)) {
  interventions = append(interventions, as.array(intervention_mat[i,])+1)
}

icpdag = dag2essgraph(as.matrix(amat), interventions)
write.table(icpdag*1, '~/Documents/causaldag/tests/icpdag1.txt', row.names=FALSE, col.names=FALSE)