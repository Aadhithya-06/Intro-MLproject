import sys

from evaluation import *

file_name = sys.argv[1]
data = np.loadtxt(file_name)

print("-" * 70)
print("10 fold cross validation on clean data")
print("-" * 70)
(acc, prec, rec, f1, confusion) = k_fold_evaluation(data)
eval_summary(acc, rec, prec, f1, confusion)
print()
print("-" * 70)
