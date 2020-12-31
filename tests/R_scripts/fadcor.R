
library('energy')
# source('/Users/chandlersquires/Desktop/energy-master/R/dcov2d.R')
library('RcppCNPy')

# args = commandArgs(trailingOnly=TRUE)
# samples_filename = args[1]
samples_filename = 'test.npy'
samples = npyLoad(samples_filename)
x = samples[, 1]
y = samples[, 2]
res = dcov2d(x, y)
print(res)