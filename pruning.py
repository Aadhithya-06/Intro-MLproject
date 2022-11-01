from copy import deepcopy
from evaluation import *
import numpy as np


def tree_depth(tree):
    if 'label' in tree:
        return 0
    return max(tree_depth(tree['right']), tree_depth(tree['left'])) + 1


def prune(data, tree):
    """ returns if the tree is pruned, original tree and pruned tree"""
    labels = data[:, -1]
    original_tree_current_node = deepcopy(tree)
    current_node = tree

    # if current_node is a leaf
    if 'label' in current_node:
        return False, original_tree_current_node, current_node

    # if current_node is a parent of 2 unchecked leaves
    if 'label' in current_node['left'] and 'label' in current_node['right'] and current_node['left']['is_checked'] \
            is False and current_node['right']['is_checked'] is False:

        left_leaf = current_node['left']
        right_leaf = current_node['right']

        left_label = left_leaf['label']
        right_label = right_leaf['label']

        left_num = np.count_nonzero(labels == left_label)
        right_num = np.count_nonzero(labels == right_label)

        # change label according to the label with the
        #  maximum occurrence of label in the training dataset
        label = left_label if left_num > right_num else right_label

        # change current node to a leaf
        current_node = {'label': label, 'is_checked': False}

        # set original tree's children is_checked to True
        original_tree_current_node['left']['is_checked'] = True
        original_tree_current_node['right']['is_checked'] = True

        return True, original_tree_current_node, current_node

    # a parent with at least one child checked
    else:
        val = current_node['value']
        attribute = current_node['attribute']
        # prune left tree
        left_data = data[np.where(data[:, attribute] < val)]
        is_modified_left, original_tree_left, modified_tree_left = prune(left_data, current_node['left'])
        # if pruning is successful, return
        if is_modified_left:
            original_tree_current_node['left'] = original_tree_left
            current_node['left'] = modified_tree_left
            return True, original_tree_current_node, current_node

        # prune right tree
        right_data = data[np.where(data[:, attribute] >= val)]
        is_modified_right, original_tree_right, modified_tree_right = prune(right_data, current_node['right'])
        if is_modified_right:
            original_tree_current_node['right'] = original_tree_right
            current_node['right'] = modified_tree_right
            return True, original_tree_current_node, current_node

    # last case: if is a parent with 2 children checked
    return False, original_tree_current_node, current_node


def k_fold_pruning(data, nr_of_folds=10):
    """returns the un-pruned measures, pruned measures and a list of all pruned trees"""
    # initiate arrays to store average measures across all folds
    all_folds_average_recall = []
    all_folds_average_precision = []
    all_folds_average_f1 = []
    all_folds_average_accuracies = []

    pruned_all_folds_average_recall = []
    pruned_all_folds_average_precision = []
    pruned_all_folds_average_f1 = []
    pruned_all_folds_average_classification = []
    pruned_trees = []

    # shuffle data to avoid it being ordered by label
    np.random.shuffle(data)
    folds_split_1 = np.split(data, nr_of_folds)

    # loop through all folds as the test data set
    for i in range(nr_of_folds):
        test_data_set = folds_split_1[i]
        training_validation_data_set = np.concatenate(folds_split_1[0:i] + folds_split_1[i + 1:])

        print('#' * 70)
        print('USING FOLD {} AS THE TEST DATA'.format(i + 1))
        print('COMPARISON MEASURE: ACCURACY')
        print('#' * 70)
        training_validation_data_set_folds = [index for index in range(nr_of_folds) if index != i]

        # initiate arrays to store results for this fold
        recall_matrix = []
        precision_matrix = []
        f1_matrix = []
        accuracies = []
        confusion_tensor = []

        pruned_recall_matrix = []
        pruned_precision_matrix = []
        pruned_f1_matrix = []
        pruned_accuracies = []
        pruned_confusion_tensor = []

        # split up dataset
        folds_split_2 = np.split(training_validation_data_set, nr_of_folds - 1)
        # loop through the remaining k-1 folds as the validation data set
        for index in range(nr_of_folds - 1):
            evaluation_data_set = folds_split_2[index]
            training_data_set = np.concatenate(folds_split_2[0:index] + folds_split_2[index + 1:])
            print('WITH FOLD {} AS THE VALIDATION DATA'
                  .format(training_validation_data_set_folds[index] + 1))
            print('TRAINING TREE ON REMAINING FOLDS...')
            # train and evaluate the un-pruned tree
            original_tree, _ = decision_tree_learning(training_data_set, 0)
            confusion = confusion_matrix(get_ground_truths(evaluation_data_set),
                                         get_predictions(evaluation_data_set, original_tree))
            acc = accuracy(confusion)
            print("Tree depth:", tree_depth(original_tree))
            print('The accuracy for the trained tree: {}'.format(acc))
            # keep a copy of the un-pruned tree
            current_tree = deepcopy(original_tree)
            # prune
            print('PRUNING TREE...')
            while True:
                # flag returns false if all possible prunes have been tested,
                # and it is not possible to prune anymore
                flag, current_tree, pruned_tree = prune(training_data_set, current_tree)

                # break if all nodes have been pruned
                if not flag:
                    break

                # evaluate pruned tree
                pruned_confusion_matrix = confusion_matrix(get_ground_truths(evaluation_data_set),
                                                           get_predictions(evaluation_data_set, pruned_tree))
                pruned_acc = accuracy(pruned_confusion_matrix)
                # check if pruned tree is better
                if pruned_acc >= acc:
                    # if better update current tree and classification rate
                    current_tree = pruned_tree
                    acc = pruned_acc

            pruned_trees.append(current_tree)
            # evaluate un-pruned and best pruned tree on validation dataset
            pruned_confusion_matrix = confusion_matrix(get_ground_truths(evaluation_data_set),
                                                       get_predictions(evaluation_data_set, current_tree))
            pruned_acc = accuracy(pruned_confusion_matrix)

            print("Pruned tree depth:", tree_depth(current_tree))
            print('The accuracy for the best pruned tree: {}'
                  .format(pruned_acc))
            print('TESTING TREES ON TEST DATA SET...')

            # evaluate un-pruned and best pruned tree on test dataset
            confusion = confusion_matrix(get_ground_truths(test_data_set),
                                         get_predictions(test_data_set, original_tree))
            acc = accuracy(confusion)
            prec = precision(confusion)
            rec = recall(confusion)
            f1 = f1_score(confusion)
            pruned_confusion_matrix = confusion_matrix(get_ground_truths(test_data_set),
                                                       get_predictions(test_data_set, current_tree))
            pruned_precision = precision(pruned_confusion_matrix)
            pruned_recall = recall(pruned_confusion_matrix)
            pruned_f1 = f1_score(pruned_confusion_matrix)
            pruned_acc = accuracy(pruned_confusion_matrix)

            print('The test accuracy for the original tree: {}'
                  .format(acc))
            print('Confusion matrix for original tree:')
            print(confusion)
            print('The test accuracy for the pruned tree: {}'
                  .format(pruned_acc))
            print('Confusion matrix for pruned tree:')
            print(pruned_confusion_matrix)
            print('-' * 70)

            accuracies.append(acc)
            pruned_accuracies.append(pruned_acc)
            # stack all label-wise measures as arrays(each fold is a row)
            # averages for each label are then the array columns averages(axis 0)
            if index == 0:
                recall_matrix = rec
                precision_matrix = prec
                f1_matrix = f1
                confusion_tensor = confusion

                pruned_recall_matrix = pruned_recall
                pruned_precision_matrix = pruned_precision
                pruned_f1_matrix = pruned_f1
                pruned_confusion_tensor = pruned_confusion_matrix
            else:
                recall_matrix = np.vstack((recall_matrix, rec))
                precision_matrix = np.vstack((precision_matrix, prec))
                f1_matrix = np.vstack((f1_matrix, f1))
                confusion_tensor = np.vstack((confusion_tensor, confusion))

                pruned_recall_matrix = np.vstack((pruned_recall_matrix, pruned_recall))
                pruned_precision_matrix = np.vstack((pruned_precision_matrix, pruned_precision))
                pruned_f1_matrix = np.vstack((pruned_f1_matrix, pruned_f1))
                pruned_confusion_tensor = np.vstack((pruned_confusion_tensor, pruned_confusion_matrix))

            print('-' * 70)

        # calculate average of evaluation measures
        average_recall = np.mean(recall_matrix, axis=0)
        average_precision = np.mean(precision_matrix, axis=0)
        average_f1 = np.mean(f1_matrix, axis=0)
        average_accuracies = np.mean(accuracies)

        pruned_average_recall = np.mean(pruned_recall_matrix, axis=0)
        pruned_average_precision = np.mean(pruned_precision_matrix, axis=0)
        pruned_average_f1 = np.mean(pruned_f1_matrix, axis=0)
        pruned_average_accuracies = np.mean(pruned_accuracies)

        # store average measures across all folds
        all_folds_average_accuracies.append(average_accuracies)
        pruned_all_folds_average_classification.append(pruned_average_accuracies)

        # stack all label-wise measures as arrays(each fold is a row)
        # averages for each label are then the array columns averages(axis 0)
        if i == 0:
            all_folds_average_recall = average_recall
            all_folds_average_precision = average_precision
            all_folds_average_f1 = average_f1

            pruned_all_folds_average_recall = pruned_average_recall
            pruned_all_folds_average_precision = pruned_average_precision
            pruned_all_folds_average_f1 = pruned_average_f1
        else:
            all_folds_average_recall = np.vstack((all_folds_average_recall, average_recall))
            all_folds_average_precision = np.vstack((all_folds_average_precision, average_precision))
            all_folds_average_f1 = np.vstack((all_folds_average_f1, average_f1))

            pruned_all_folds_average_recall = np.vstack((pruned_all_folds_average_recall, pruned_average_recall))
            pruned_all_folds_average_precision = np.vstack((pruned_all_folds_average_precision,
                                                            pruned_average_precision))
            pruned_all_folds_average_f1 = np.vstack((pruned_all_folds_average_f1, pruned_average_f1))

    # calculate average of evaluation measures across all folds
    average_recall = np.mean(all_folds_average_recall, axis=0)
    average_precision = np.mean(all_folds_average_precision, axis=0)
    average_f1 = np.mean(all_folds_average_f1, axis=0)
    average_accuracies = np.mean(all_folds_average_accuracies)

    pruned_average_recall = np.mean(pruned_all_folds_average_recall, axis=0)
    pruned_average_precision = np.mean(pruned_all_folds_average_precision, axis=0)
    pruned_average_f1 = np.mean(pruned_all_folds_average_f1, axis=0)
    pruned_average_accuracies = np.mean(pruned_all_folds_average_classification)

    eval_measures = [average_accuracies, average_recall, average_precision, average_f1]
    pruned_eval_measures = [pruned_average_accuracies, pruned_average_recall, pruned_average_precision,
                            pruned_average_f1]
    improvement = pruned_average_accuracies - average_accuracies

    print('Average accuracy for un-pruned trees: {}'.format(average_accuracies))
    print('Average accuracy for pruned trees: {}'.format(pruned_average_accuracies))
    print('Pruning improved the average test score by {}%' .format(improvement * 100))
    print('Average recall for un-pruned trees: {}'
          .format(average_recall))
    print('Average precision for un-pruned trees: {}'
          .format(average_precision))
    print('Average F1 score for un-pruned trees: {}'
          .format(average_f1, 3))
    print('Average recall for pruned trees: {}'
          .format(pruned_average_recall))
    print('Average precision for pruned trees: {}'
          .format(pruned_average_precision))
    print('Average F1 score for pruned trees: {}'
          .format(pruned_average_f1))
    return eval_measures, pruned_eval_measures
