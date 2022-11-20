import argparse

import os
import sys

from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='clf_name and random_state', prefix_chars='--')

# Add the arguments
parser.add_argument('clf_name', metavar='clf_name', type=str, help='clf_name')
parser.add_argument('random_state', metavar='random_state', type=int, help='random_state')

args = parser.parse_args()

clf_name = args.clf_name
random_state = args.random_state

print(clf_name)
print(random_state)

if clf_name == 'svm':
    clf = svm.SVC()
else:
    clf = DecisionTreeClassifier()

digits = datasets.load_digits()

h_params = None

if clf_name == 'svm':
    h_params = {'gamma': 0.02, 'C': 0.1}
else:
    h_params = {'max_depth': 3}

train_f = 0.8
test_f = 0.1
dev_f = 0.1

imgs = digits.images
target = digits.target

n_samples = len(imgs)
data = imgs.reshape((n_samples, -1))

dev_test_f = 1 - train_f
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, target, test_size=dev_test_f, shuffle=True, random_state=random_state
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_f) / dev_test_f, shuffle=True, random_state=random_state
)

clf = None
if clf_name == 'svm':
    clf = svm.SVC()
else:
    clf = DecisionTreeClassifier()

clf.set_params(**h_params)

clf.fit(X_train, y_train)

predicted_dev = clf.predict(X_dev)
predicted_train = clf.predict(X_train)
predicted_test = clf.predict(X_test)

cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

print("Accuracy: " + str(cur_acc))
print("Train: " + str(cur_train_acc))
print("Test: " + str(cur_test_acc))