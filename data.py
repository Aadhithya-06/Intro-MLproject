import numpy as np


def import_clean_data():
    """returns a numpy array of clean_dataset"""
    return np.loadtxt('./wifi_db/clean_dataset.txt')


def import_noisy_data():
    """returns a numpy array of noisy_dataset"""
    return np.loadtxt('./wifi_db/noisy_dataset.txt')
