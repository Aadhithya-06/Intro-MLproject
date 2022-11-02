from plot_tree import *
from pruning import *

if __name__ == '__main__':
    data = import_clean_data()
    res = decision_tree_learning(data)

    k_fold_pruning(data)
