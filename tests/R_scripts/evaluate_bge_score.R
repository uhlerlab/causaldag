library("BiDAG")
library("RcppCNPy")

samples = npyLoad("tests/data/bge_data/samples.npy")
myScore = scoreparameters(
  ncol(samples),
  "bge",
  samples,
  weightvector = NULL,
  bgnodes=c(),
  bgepar = list(am = ncol(samples), aw = NULL)
)
dag_amat = npyLoad("tests/data/bge_data/dag_amat.npy", type="integer")
n = ncol(dag_amat)
dag_amat = diag(n) 
dag_amat[upper.tri(dag_amat, diag = FALSE)] = 1
dag_amat[lower.tri(dag_amat, diag = TRUE)] = 0
# dag_amat[diag(n)] = 0
# print(dag_amat)
score = DAGscore(myScore, dag_amat)
print(score)
npySave("tests/data/bge_data/r_bge.npy", score)