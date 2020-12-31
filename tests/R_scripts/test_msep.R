#!/usr/bin/env Rscript
suppressMessages(library(ggm))
suppressMessages(library(tseries))
# source('./config')

args = commandArgs(trailingOnly=TRUE)
adjacency_matrix_filename = args[1]
nodes1 = as.numeric(unlist(strsplit(args[2], ','))) + 1
nodes2 = as.numeric(unlist(strsplit(args[3], ','))) + 1
cond_set = if (length(args) == 4) as.numeric(unlist(strsplit(args[4], ','))) + 1 else c()
adjacency_matrix = as.matrix(read.table(adjacency_matrix_filename))
# row.names(adjacency_matrix) = 1:dim(adjacency_matrix)[1]
# print(adjacency_matrix)

res = msep(adjacency_matrix, nodes1, nodes2, cond_set)
cat(res)