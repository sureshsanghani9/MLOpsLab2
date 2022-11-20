import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def get_train_test_split(data, seed, imgs, target):
    train_f = 0.8
    test_f = 0.1
    dev_f = 0.1

    n_samples = len(imgs)
    data = imgs.reshape((n_samples, -1))

    dev_test_f = 1 - train_f
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, target, test_size=dev_test_f, shuffle=True, random_state=seed
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_f) / dev_test_f, shuffle=True, random_state=seed
    )
    return X_train, X_dev_test, y_train, y_dev_test, X_test, X_dev, y_test, y_dev