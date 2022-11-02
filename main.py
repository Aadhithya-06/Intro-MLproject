from plot_tree import *
from pruning import *

clean_data = import_clean_data()
noisy_data = import_noisy_data()


def evaluation():
    print("-" * 70)
    print("10 fold cross validation on clean data")
    print("-" * 70)
    (acc, prec, rec, f1, confusion) = k_fold_evaluation(clean_data)
    eval_summary(acc, rec, prec, f1, confusion)
    print()
    print("-" * 70)
    print()
    print("-" * 70)
    print("10 fold cross validation on noisy data")
    print("-" * 70)
    (acc, prec, rec, f1, confusion) = k_fold_evaluation(noisy_data)
    eval_summary(acc, rec, prec, f1, confusion)
    print()
    print("-" * 70)


def pruning_evaluation():
    print("-" * 70)
    print("Pruning and 10 fold cross validation on clean data")
    print("-" * 70)
    k_fold_pruning(clean_data)
    print()
    print("-" * 70)
    print()
    print("-" * 70)
    print("Pruning and 10 fold cross validation on noisy data")
    print("-" * 70)
    k_fold_pruning(noisy_data)
    print()
    print("-" * 70)


if __name__ == '__main__':
    evaluation()
    pruning_evaluation()
