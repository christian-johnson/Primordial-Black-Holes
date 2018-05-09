import pickle
import numpy as np

file = open('results.pk1')
g = pickle.load(file)
file.close()
i = 0
arr_of_candidates = []
x = np.zeros((len(g)))
for entry in g:
    if entry[0] not in arr_of_candidates:
        arr_of_candidates.append(entry[0])
        x[i] = entry[1]
        i += 1
x = x[np.nonzero(x)]
print "num of candidates = " + str(len(x))
print "Standard deviation = " + str(np.std(x))
print "Max entries:" 
for entry in g:
    if entry[1]>np.sort(x)[::-1][10]:
	print entry
import matplotlib.pyplot as plt
bins = np.linspace(-5.0, 5.0, 50)
plt.hist(x,bins)
plt.show()



