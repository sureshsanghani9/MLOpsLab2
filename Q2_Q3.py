import argparse

import os
import sys

from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import f1_score
from joblib import dump

parser = argparse.ArgumentParser(description='clf_name and random_state', prefix_chars='--')

# Add the arguments
parser.add_argument('clf_name', metavar='clf_name', type=str, help='clf_name')
parser.add_argument('random_state', metavar='random_state', type=int, help='random_state')

args = parser.parse_args()

clf_name = args.clf_name
random_state = args.random_state

#print(clf_name)
#print(random_state)

if clf_name == 'svm':
    clf = svm.SVC()
else:
    clf = DecisionTreeClassifier()

digits = datasets.load_digits()

h_params = None
gamma_lst = [0.02, 0.007, 0.003, 0.0009, 0.0001, 0.0006]
c_lst = [0.1, 0.3, 0.8, 0.7, 2, 0.4]
d_lst = [3, 4, 5, 6, 8]

g = random.choice(gamma_lst)
c = random.choice(c_lst)
d = random.choice(d_lst)

if clf_name == 'svm':
    h_params = {'gamma': g, 'C': c}
else:
    h_params = {'max_depth': d}

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

"""
cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

print("Accuracy: " + str(cur_acc))
print("Train: " + str(cur_train_acc))
print("Test: " + str(cur_test_acc))
"""

test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
test_macro = f1_score(y_test, predicted_test, average='macro')
file_name  = ''
res_file = str(clf_name + '_' + str(random_state) + '.txt')
if clf_name == 'svm':
    file_name = 'svm_gamma=' + str(g) + '_C=' + str(c) + '.joblib'
else:
    file_name = 'tree_max_depth=' + str(d) + '.joblib'

file_cont = "test accuracy: " + str(test_acc) + " \n test macro-f1: "+ str(test_macro) +" \n model saved at ./models/"+ file_name +""

with open('results/'+res_file, 'w') as the_file:
    the_file.write(file_cont)

dump(clf, 'models/'+ file_name)