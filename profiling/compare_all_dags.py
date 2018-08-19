import causaldag as cd
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1729)

all_dags_times = np.zeros(50)
all_dags2_times = np.zeros(50)
dags = cd.rand.directed_erdos(8, .5, 50)
for i, dag in enumerate(dags):
    print(i)
    cpdag = dag.cpdag()

    start = time.time()
    all_dags = cpdag.all_dags()
    all_dags_times[i] = time.time() - start

    start = time.time()
    all_dags2 = cpdag.all_dags2()
    all_dags2_times[i] = time.time() - start

    if not all_dags == all_dags2:
        raise Exception('Not equal')


