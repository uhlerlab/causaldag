import causaldag as cd
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1729)
dags = cd.rand.directed_erdos(8, .5, 50)
cpdags = [dag.cpdag() for dag in dags]
arcs = np.array([len(dag.arcs) for dag in dags])
dir_arcs = np.array([len(cpdag.arcs) for cpdag in cpdags])
edges = np.array([len(cpdag.edges) for cpdag in cpdags])


def run_all_dags():
    all_dags_times = np.zeros(len(dags))
    all_dags_list = []
    for i, cpdag in enumerate(cpdags):
        print(i)

        start = time.time()
        all_dags = cpdag.all_dags()
        all_dags_times[i] = time.time() - start
        all_dags_list.append(all_dags)
    return all_dags_times, all_dags_list


def run_all_dags2():
    all_dags2_times = np.zeros(len(dags))
    all_dags_list = []
    for i, cpdag in enumerate(cpdags):
        print(i)

        start = time.time()
        all_dags = cpdag.all_dags2()
        all_dags2_times[i] = time.time() - start
        all_dags_list.append(all_dags)
    return all_dags2_times, all_dags_list


times1, all_dags = run_all_dags()
times2, all_dags2 = run_all_dags2()
print(all(d1 == d2 for d1, d2 in zip(all_dags, all_dags2)))

mean1 = times1.mean()
mean2 = times2.mean()

median1 = np.median(times1)
median2 = np.median(times2)