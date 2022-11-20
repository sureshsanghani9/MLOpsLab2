import sys, os
import numpy as np
from joblib import load
from sklearn import datasets, svm, metrics


sys.path.append(".")

from utils import get_train_test_split

def test_data_split_same_when_seed_same():
    digits = datasets.load_digits()
    X_train, X_dev_test, y_train, y_dev_test, X_test, X_dev, y_test, y_dev = get_train_test_split(digits, 42, digits.images, digits.target)
    X_train1, X_dev_test1, y_train1, y_dev_test1, X_test1, X_dev1, y_test1, y_dev1 = get_train_test_split(digits, 42, digits.images, digits.target)
    print(X_train.shape)
    print(X_train1.shape)
    assert np.array_equal(X_train, X_train1)
    assert np.array_equal(X_dev_test, X_dev_test1)
    assert np.array_equal(y_train, y_train1)
    assert np.array_equal(y_dev_test, y_dev_test1)
    assert np.array_equal(X_test, X_test1)
    assert np.array_equal(X_dev, X_dev1)
    assert np.array_equal(y_test, y_test1)
    assert np.array_equal(y_dev, y_dev1)

def test_data_split_different_when_seed_different():
    digits = datasets.load_digits()
    X_train, X_dev_test, y_train, y_dev_test, X_test, X_dev, y_test, y_dev = get_train_test_split(digits, 42, digits.images, digits.target)
    X_train1, X_dev_test1, y_train1, y_dev_test1, X_test1, X_dev1, y_test1, y_dev1 = get_train_test_split(digits, 55, digits.images, digits.target)
    print(X_train.shape)
    print(X_train1.shape)
    assert np.array_equal(X_train, X_train1) == False
    assert np.array_equal(X_dev_test, X_dev_test1) == False
    assert np.array_equal(y_train, y_train1) == False
    assert np.array_equal(y_dev_test, y_dev_test1) == False
    assert np.array_equal(X_test, X_test1) == False
    assert np.array_equal(X_dev, X_dev1) == False
    assert np.array_equal(y_test, y_test1) == False
    assert np.array_equal(y_dev, y_dev1) == False
