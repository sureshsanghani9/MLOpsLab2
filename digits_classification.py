import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np


digits = datasets.load_digits()

def run_with_hyperparameters(imgs, target, res):
    gamma_lst = [0.02, 0.007, 0.003, 0.0009, 0.0001, 0.0006]
    c_lst = [0.1, 0.3, 0.8, 0.7, 2, 0.4] 

    h_params = [{'gamma':g, 'C':c} for g in gamma_lst for c in c_lst]

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

        clf = svm.SVC()

        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        clf.fit(X_train, y_train)

        predicted_dev = clf.predict(X_dev)
        predicted_train = clf.predict(X_train)
        predicted_test = clf.predict(X_test)

        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
        cur_train_acc = metrics.accuracy_score(y_pred=predicted_train, y_true=y_train)
        cur_test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)

        h_param_results.append([hyper_params['gamma'], hyper_params['C'], cur_acc, cur_train_acc, cur_test_acc])
        if cur_acc > best_accuracy:
            best_accuracy = cur_acc
            best_dev_accuracy = cur_acc
            best_train_accuracy = cur_train_acc
            best_test_accuracy = cur_test_acc
            best_model = clf
            best_h_params = cur_h_params

    df_h_param_results = pd.DataFrame(h_param_results, columns =['Gamma', 'C', 'Dev_Accuracy', 'Train_Accuracy', 'Test_Accuracy'])
    print("\n================================Hyperparameter and Results==================================================")
    print(df_h_param_results.head(10))

    predicted = best_model.predict(X_test)

    print("\n")    
    print("\n================================Report=====================================================================")
    print(
        f"Classification report for classifier {best_model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    print("\n") 
    print("\n================================Best hyperparameters and Accuracy===========================================")
    print("Best C:" + str(best_h_params['C']) + " and Gamma:" + str(best_h_params['gamma']) )
    print("Best Dev Accuracy:" + str(best_dev_accuracy))
    print("Best Train Accuracy:" + str(best_train_accuracy))
    print("Best Test Accuracy:" + str(best_test_accuracy))

    print("\n") 
    print("\n================================Predicted Images=============================================================")
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(res, res)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        

run_with_hyperparameters(digits.images, digits.target, res=8)


def resize_images(res, imgs):
    print("\n================================Current Image size=============================================================")
    print("Current Image size: " + str(digits.images[0].shape))
    print("\n================================New Image size=============================================================")
    print("Image Resolution: " + '(' + str(res) + ',' + str(res) + ')')
    arr = []
    for im in imgs:
        image_resized = resize(im, (res, res), anti_aliasing=True)
        arr.append(image_resized)
    return np.array(arr)
    

#new resolution 6*6
res = 6
run_with_hyperparameters(resize_images(res,digits.images), digits.target, res)


#new resolution 12*12
res = 12
run_with_hyperparameters(resize_images(res,digits.images), digits.target, res)



#new resolution 18*18
res = 18
run_with_hyperparameters(resize_images(res,digits.images), digits.target, res)