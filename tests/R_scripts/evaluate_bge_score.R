library("BiDAG")
library("RcppCNPy")

samples = npyLoad("tests/data/bge_data/samples.npy")
myScore = scoreparameters(
  ncol(samples),
  "bge",
  samples,
  weightvector = NULL,
  bgnodes=c()
)
dag_amat = npyLoad("tests/data/bge_data/dag_amat.npy")
score = DAGscore(myScore, dag_amat)
npySave("tests/data/bge_data/r_bge.npy", score)
