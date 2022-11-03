import sys

from pruning import *

file_name = sys.argv[1]
data = np.loadtxt(file_name)

print("-" * 70)
print("Pruning and 10 fold cross validation on clean data")
print("-" * 70)
k_fold_pruning(data)
print()
print("-" * 70)
