import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.tree import DecisionTreeClassifier
import numpy as np

digits = datasets.load_digits()

def run_with_hyperparameters(imgs, target, clf_name, iteraction):
    gamma_lst = [0.02, 0.007, 0.003, 0.0009, 0.0001, 0.0006]
    c_lst = [0.1, 0.3, 0.8, 0.7, 2, 0.4] 

    h_params = []
    
    if clf_name == 'SVM':
        h_params = [{'gamma':g, 'C':c} for g,c in zip(gamma_lst,c_lst)]
    else:
        h_params = [{'max_depth': m} for m in [None, 3, 4, 5, 6, 8]]

    train_f = 0.8
    test_f = 0.1
    dev_f = 0.1

    n_samples = len(imgs)
    data = imgs.reshape((n_samples, -1))

    dev_test_f = 1-train_f
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, target, test_size=dev_test_f, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_f)/dev_test_f, shuffle=True
    )


    best_accuracy = -1.0
    best_model = None
    best_h_params = None

    best_train_accuracy = -1.0
    best_dev_accuracy = -1.0
    best_test_accuracy = -1.0

    h_param_results = []
    for cur_h_params in h_params:
        clf, hyper_params = None, None
        
        if clf_name == 'SVM':
            clf = svm.SVC()
        else:
            clf = DecisionTreeClassifier()
        

        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        clf.fit(X_train, y_train)

        predicted_dev = clf.predict(X_dev)
        predicted_train = clf.predict(X_train)
        predicted_test = clf.predict(X_test)

        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
        cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
        cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

        if clf_name == 'SVM':
            h_param_results.append([hyper_params['gamma'], hyper_params['C'], cur_acc, cur_train_acc, cur_test_acc])
        else:
            h_param_results.append([hyper_params['max_depth'], cur_acc, cur_train_acc, cur_test_acc])
            
        if cur_acc > best_accuracy:
            best_accuracy = cur_acc
            best_dev_accuracy = cur_acc
            best_train_accuracy = cur_train_acc
            best_test_accuracy = cur_test_acc
            best_model = clf
            best_h_params = cur_h_params

    if clf_name == 'SVM':
        df_h_param_results = pd.DataFrame(h_param_results, columns =['Gamma', 'C', 'Dev_Accuracy', 'Train_Accuracy', 'Test_Accuracy'])
    else:
        df_h_param_results = pd.DataFrame(h_param_results, columns =['Max_Depth', 'Dev_Accuracy', 'Train_Accuracy', 'Test_Accuracy'])
    
    print(f"\n===================Hyperparameter and Results for {clf_name} for iteration {str(iteraction)}=======================")
    print(df_h_param_results.head(10))

    predicted = best_model.predict(X_test)

    """
    print("\n")    
    print("\n================================Report=====================================================================")
    print(
        f"Classification report for {clf_name} classifier {best_model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n")
    """
    
    print(f"\n===============Best hyperparameters and Accuracy for {clf_name} for iteration {str(iteraction)}====================")
    if clf_name == 'SVM':
        print("Best C:\t\t\t\t\t" + str(best_h_params['C']) + " and Gamma:" + str(best_h_params['gamma']) )
    else:
        print("Best Max_Depth:\t\t\t\t" + str(best_h_params['max_depth']))
        
    print("Best Dev Accuracy:\t\t\t" + str(best_dev_accuracy))
    print("Best Train Accuracy:\t\t\t" + str(best_train_accuracy))
    print("Best Test Accuracy:\t\t\t" + str(best_test_accuracy))

    no_of_correct_pred = len([k for k, (a, b) in enumerate(zip(y_test, predicted)) if a == b])
    print("Best No of Correct Prediction:\t\t" + str(no_of_correct_pred))
    print("\n") 
    
    return best_test_accuracy, no_of_correct_pred


t = 5
SVM_accu = []
DT_accu = []
SVM_correct_count = []
DT_correct_count = []

for i in range(0,t):
    acc, cc = run_with_hyperparameters(digits.images, digits.target, 'SVM', i)
    SVM_accu.append(acc)
    SVM_correct_count.append(cc)

for i in range(0,t):
    acc, cc = run_with_hyperparameters(digits.images, digits.target, 'Decision Tree', i)
    DT_accu.append(acc)
    DT_correct_count.append(cc)

print("\n================================SVM vs Decision Tree Accuracy comparision===================================")
print("\nRun\t\t\tSVM\t\t\tDecision Tree")
for i in range(0,t):
    print(f"\n{str(i)}:\t\t\t{str(round(SVM_accu[i],4))}\t\t\t{str(round(DT_accu[i],4))}")
    
print(f"\nMean:\t\t\t{str(round(np.mean(SVM_accu),4))}\t\t\t{str(round(np.mean(DT_accu),4))}")
                      
print(f"\nSD:\t\t\t{str(round(np.std(SVM_accu),4))}\t\t\t{str(round(np.std(DT_accu),4))}")


print("\n")
print("\n======================SVM vs Decision Tree Number of currect prediction comparision========================")
print("\nRun\t\t\tSVM\t\t\tDecision Tree")
for i in range(0,t):
    print(f"\n{str(i)}:\t\t\t{str(round(SVM_correct_count[i],4))}\t\t\t{str(round(DT_correct_count[i],4))}")
    
print(f"\nMean:\t\t\t{str(round(np.mean(SVM_correct_count),4))}\t\t\t{str(round(np.mean(DT_correct_count),4))}")
                      
print(f"\nSD:\t\t\t{str(round(np.std(SVM_correct_count),4))}\t\t\t{str(round(np.std(DT_correct_count),4))}")