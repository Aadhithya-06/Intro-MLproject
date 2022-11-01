from data import *


def h(labels):
    """ returns the H-function output of labels"""
    total = len(labels)
    _, no_occurences = np.unique(labels, return_counts=True)
    prob = no_occurences / total

    return -sum(prob * np.log2(prob))


def gain(data, left_data, right_data):
    """ returns the gain value of an attribute according to left and right data"""
    data_size = len(data)
    left_data_size = len(left_data)
    right_data_size = len(right_data)

    remainder = (h(left_data) * left_data_size + h(right_data) * right_data_size) / data_size

    return h(data) - remainder


def attribute_gain(data, index):
    """ returns a tuple of maximum gain and value to split an attribute according to its gain"""

    # iterate through each value in attribute and compute gain
    optimal_attr_split = (-float('inf'), 0, [], [])

    sorted_data = data[data[:, index].argsort()]  # sort data by attribute
    attr_values = np.unique(data[:, index])

    # split data to left and right, and find the best att_index to split
    for val in attr_values:
        left_data = sorted_data[np.where(sorted_data[:, index] < val)]
        right_data = sorted_data[np.where(sorted_data[:, index] >= val)]

        attr_gain = gain(sorted_data, left_data, right_data)

        if attr_gain > optimal_attr_split[0]:
            optimal_attr_split = (attr_gain, val, left_data, right_data)

    return optimal_attr_split


def find_split(data):
    """ returns the optimal attribute, value, left data and right data"""
    optimal_gain = -float('inf')
    optimal_split = (0, 0, [], [])
    # iterate through all attributes to find the optimal attribute to split
    for attr in range(len(data[0]) - 1):
        g, value, left_data, right_data = attribute_gain(data, attr)
        if g > optimal_gain:
            optimal_gain = g
            optimal_split = (attr, value, left_data, right_data)

    return optimal_split


def decision_tree_learning(dataset, depth=0):
    """ return: root node of the decision tree """
    # check if all values in the label are the same
    if len(np.unique(dataset[:, -1])) == 1:
        return {'label': int(dataset[:, -1][0]), 'is_checked': False}, depth

    else:
        # find attribute and value to split the dataset
        attribute, value, left_data, right_data = find_split(dataset)

        # recursively calls the function on left and right node
        left_node, left_depth = decision_tree_learning(left_data, depth + 1)
        right_node, right_depth = decision_tree_learning(right_data, depth + 1)

        root = {'attribute': attribute, 'value': value, 'left': left_node, 'right': right_node}

        return root, max(left_depth, right_depth)
