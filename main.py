from decision_tree_algo import *
from plot_tree import *
from pruning import *

if __name__ == '__main__':
    data = import_clean_data()
    res = decision_tree_learning(data)

    print(res[1])

    (acc, prec, rec, f1, confusion) = k_fold_evaluation(data)

    eval_summary(acc, rec, prec, f1, confusion)



