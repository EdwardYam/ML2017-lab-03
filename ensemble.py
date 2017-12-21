import pickle
import numpy as np
import math


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit


    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        weight = 1 / X.shape[0] * np.ones((1, X.shape[0]))
        clf_list = []
        a_list = []
        error_list = []
        for i in range(0, self.n_weakers_limit):
            #clf = self.weak_classifier(random_state=0)
            clf = self.weak_classifier(max_depth=2)
            clf.fit(X, y, sample_weight=weight[0])
            clf_list.append(clf)

            y_ = clf.predict(X)

            error_rate = np.sum(weight * (y_!= y))
            error_list.append(error_rate)
            if(error_rate > 0.5):
                break

            a = 0.5 * math.log((1-error_rate) / error_rate)
            z = np.sum(weight * np.power(math.e, -a * y * y_))
            weight = (weight / z) * np.power(math.e, -a * y * y_)
            a_list.append(a)

        a_list = np.array(a_list)
        model = (clf_list, a_list)
        #print(error_list)
        self.save(model, "ada_model.m")


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        result = np.zeros((self.n_weakers_limit, X.shape[0]))
        clf_list, a_list = self.load("ada_model.m")

        for i in range(self.n_weakers_limit):
            result[i] = clf_list[i].predict(X)
        a_list = np.array(a_list)
        #print(a_list)
        predict_result = (np.mean(result * a_list.reshape(-1, 1), axis=0))
        predict_result[predict_result < threshold] = -1
        predict_result[predict_result >= threshold] = 1

        return predict_result



    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
