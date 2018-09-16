import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import utils
import time


def get_knn_score(train_data, test_data):
    features = np.array([utils.hog_features(img) for img in train_data[:, :-1]])
    labels = train_data[:, -1]
    k = 13
    print("Fitting KNN model with k =", k)
    tik = time.clock()
    knn_clf = KNeighborsClassifier(n_neighbors=k).fit(features, labels)
    print("Time taken to fit = {:.4} sec".format(float(time.clock() - tik)))

    print("Predicting using KNN with k =", k)
    test_features = np.array([utils.hog_features(img) for img in test_data[:, :-1]])
    test_labels = test_data[:, -1]
    tik = time.clock()
    score = knn_clf.score(test_features, test_labels)
    print("Time taken to predict = {:.4} sec".format(float(time.clock() - tik)))
    return score


def get_svm_score(train_data, test_data):
    features = np.array([utils.hog_features(img) for img in train_data[:, :-1]])
    labels = train_data[:, -1]
    c = 15000
    print("Fitting SVM model with C =", c)
    tik = time.clock()
    svm_clf = SVC(C=c).fit(features, labels)
    print("Time taken to fit = {:.4} sec".format(float(time.clock() - tik)))

    print("Predicting using SVM with C =", c)
    test_features = np.array([utils.hog_features(img) for img in test_data[:, :-1]])
    tik = time.clock()
    test_labels = test_data[:, -1]
    score = svm_clf.score(test_features, test_labels)
    print("Time taken to predict = {:.4} sec".format(float(time.clock() - tik)))
    return score


train_data, test_data = utils.read_mnist()
# print("KNN score =", get_knn_score(train_data, test_data))
print("SVM score =", get_svm_score(train_data, test_data))
