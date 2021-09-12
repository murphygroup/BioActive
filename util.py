# Useful library for handling iterators
import itertools

import numpy as np

def createTuples(arg):
	ranges = []
	for i in range(len(arg)):
		ranges.append(list(range(arg[i][1], arg[i][2] + 1, 1)))
	result = list(itertools.product(*ranges))
	return np.asarray(result)