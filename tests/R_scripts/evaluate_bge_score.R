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
dag_amat = npyLoad("tests/data/bge_data/dag_amat.npy")
dag_amat = 1*(dag_amat>0)
score = DAGscore(myScore, dag_amat)
npySave("tests/data/bge_data/r_bge.npy", score)
