from decision_tree_algo import decision_tree_learning
import numpy as np


def get_predictions(dataset, tree):
    """returns a list of predictions made by the decision tree model"""

    predicted = []
    for sample in dataset:
        current_node = tree
        attribute_values = sample[:-1]

        while True:
            # if we find the leaf node
            # add it to the list
            if 'label' in current_node:
                predicted_label = current_node['label']
                predicted.append(predicted_label)
                break
            # iterate through left node if attribute value is less than the split node value
            elif attribute_values[current_node['attribute']] < current_node['value']:
                current_node = current_node['left']
            # iterate through right node if attribute value is not less than the split node value
            else:
                current_node = current_node['right']

    return np.array(predicted)


def get_ground_truths(dataset):
    t = []
    for i in dataset[:, -1]:
        t.append(int(i))
    return np.array(t)


def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)

        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def accuracy(confusion):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """

    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.


def precision(confusion):
    """ Compute the precision score per class given the ground truth and predictions

    Also return the macro-averaged precision across classes.

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
            Rows are ground truth per class, columns are predictions

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the
              precision for class c
            - macro-precision is macro-averaged precision (a float)
    """
    p = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    return p


def f1_score(confusion):
    """ Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
            Rows are ground truth per class, columns are predictions

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    precisions = precision(confusion)
    recalls = recall(confusion)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    return f


def recall(confusion):
    """ Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
            Rows are ground truth per class, columns are predictions

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the
                recall for class c
            - macro-recall is macro-averaged recall (a float)
    """
    r = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    return r


def k_fold_evaluation(data, nr_of_folds=10, shuffle=True):
    """
    return: average confusion matrix, recall, precision, F1 score,
     accuracy across the K-folds.
    """
    # shuffle the data so it is not in labeled order
    if shuffle:
        np.random.shuffle(data)

    folds = np.split(data, nr_of_folds)

    # initiate arrays for storing evaluation measures
    recall_matrix = []
    precision_matrix = []
    f1_matrix = []
    accuracies = []
    confusion_tensor = []

    for index in range(nr_of_folds):
        print('USING FOLD {} AS THE VALIDATION DATA'.format(index + 1))
        # pick out folds for training and testing
        test_data_set = folds[index]
        training_data_set = np.concatenate(folds[0:index] + folds[index + 1:])

        # train the tree
        tree, _ = decision_tree_learning(training_data_set, 0)
        # evaluate the tree
        confusion = confusion_matrix(get_ground_truths(test_data_set), get_predictions(test_data_set, tree))
        acc = accuracy(confusion)
        prec = precision(confusion)
        rec = recall(confusion)
        f1 = f1_score(confusion)

        print('-' * 70)
        # store evaluation measures
        confusion = np.reshape(confusion, (1, 4, 4))
        accuracies.append(acc)

        if index == 0:
            recall_matrix = rec
            precision_matrix = prec
            f1_matrix = f1
            confusion_tensor = confusion
        else:
            recall_matrix = np.vstack((recall_matrix, rec))
            precision_matrix = np.vstack((precision_matrix, prec))
            f1_matrix = np.vstack((f1_matrix, f1))
            confusion_tensor = np.vstack((confusion_tensor, confusion))

    # calculate mean of evaluation measures
    average_recall = np.mean(recall_matrix, axis=0)
    average_precision = np.mean(precision_matrix, axis=0)
    average_f1 = np.mean(f1_matrix, axis=0)
    average_accuracy = np.mean(accuracies)
    average_confusion_matrix = np.mean(confusion_tensor, axis=0)

    return (average_accuracy,
            average_recall,
            average_precision,
            average_f1,
            average_confusion_matrix)


def eval_summary(average_accuracy, average_recall, average_precision, average_f1,
                 average_confusion_matrix):
    print('The averages of the evaluation metrics are as follows:')
    print("\nconfusion matrix:\n {}".format(average_confusion_matrix))
    print("\naccuracy: {}".format(average_accuracy))
    print("\nprecision: {}".format(average_precision))
    print("\nrecall: {}".format(average_recall))
    print("\nf1: {}".format(average_f1))
