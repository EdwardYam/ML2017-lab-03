#coding: utf-8
import pickle
import numpy as np
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import ensemble
import process

#Read the data and split it into training set and test set
def get_data():

    percent = 0.8

    with open("face.data", 'rb') as file:
        face_data = pickle.load(file)
    with open("nonface.data", 'rb') as file:
        nonface_data = pickle.load(file)

    data = np.concatenate((face_data, nonface_data), axis=0)
    data = list(data)

    random.shuffle(data)
    data = np.array(data)
    train_size = int(data.shape[0] * percent)
    train = data[0: train_size, :]
    test =  data[train_size: , :]

    x_train = train[:, 0: -1]
    y_train = train[:, -1]

    x_test = test[:, 0: -1]
    y_test = test[:, -1]

    return x_train, y_train, x_test, y_test

#Begin to train
if __name__ == "__main__":
    process.process_data()

    x_train, y_train, x_test, y_test = get_data()

    adaBoost = ensemble.AdaBoostClassifier(DecisionTreeClassifier, 1)
    adaBoost.fit(x_train, y_train)
    ytest_ = adaBoost.predict(x_test)
    print(classification_report(y_test, ytest_))

    adaBoost = ensemble.AdaBoostClassifier(DecisionTreeClassifier, 2)
    adaBoost.fit(x_train, y_train)
    ytest_ = adaBoost.predict(x_test)
    print(classification_report(y_test, ytest_))

    adaBoost = ensemble.AdaBoostClassifier(DecisionTreeClassifier, 5)
    adaBoost.fit(x_train, y_train)
    ytest_ = adaBoost.predict(x_test)
    print(classification_report(y_test, ytest_))

    adaBoost = ensemble.AdaBoostClassifier(DecisionTreeClassifier, 10)
    adaBoost.fit(x_train, y_train)
    ytest_ = adaBoost.predict(x_test)
    print(classification_report(y_test, ytest_))

    # array1 = np.array([
    #     [1, 2, 3]
    # ])
    #
    # array2 = np.array([
    #     [2, 2, 3]
    # ])
    #
    # array3 = np.array([
    #     [2, 2, 2]
    # ])
    #
    # print(array3 * (array1 != array2))