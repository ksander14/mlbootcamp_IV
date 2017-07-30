
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def one_classifier_voting_stacking(X_train, y_train, X_test, classifier, n_folds=5, test_mean=True):
    skf_cv = StratifiedKFold(n_splits=n_folds, random_state=0)
    new_x_train = np.array([0 for i in range(len(X_train))])
    new_x_test = np.array([[0 for i in range(len(X_test))] for i in range(n_folds)])
    i = 0
    mean_accur = 0
    for train_index, test_index in skf_cv.split(X_train, y_train):
        X_train_cur, X_test_cur = X_train.as_matrix()[train_index], X_train.as_matrix()[test_index]
        y_train_cur, y_test_cur = y_train.as_matrix()[train_index], y_train.as_matrix()[test_index]
        classifier.fit(X_train_cur, y_train_cur)
        new_x_train[test_index] = classifier.predict(X_test_cur)
        new_x_test[i] = classifier.predict(X_test.as_matrix())
        i += 1
        mean_accur += accuracy_score(new_x_train[test_index], y_test_cur)
    mean_accur /= n_folds
    new_x_test_final = np.array([0 for i in range(len(X_test))])
    if test_mean:
        for i in range(len(new_x_test[0])):
            new_x_test_final[i] = np.bincount(new_x_test[:, i]).argmax()
    else:
        classifier.fit(X_train, y_train)
        new_x_test_final = classifier.predict(X_test)
    return new_x_train, new_x_test_final, mean_accur


# In[39]:

def one_classifier_proba_stacking(X_train, y_train, X_test, classifier, num_classes, n_folds=5, test_mean=True):
    skf_cv = StratifiedKFold(n_splits=n_folds, random_state=42)    
    X_train_new = np.array([[0.0 for i in range(num_classes)] for i in range(len(X_train))])
    X_test_new = np.array([[[0.0 for i in range(num_classes)] for i in range(len(X_test))] for i in range(n_folds)])
    i = 0
    mean_accur = 0
    for train_index, test_index in skf_cv.split(X_train, y_train):
        X_train_cur, X_test_cur = X_train.as_matrix()[train_index], X_train.as_matrix()[test_index]
        y_train_cur, y_test_cur = y_train.as_matrix()[train_index], y_train.as_matrix()[test_index]
        classifier.fit(X_train_cur, y_train_cur)
        X_train_new[test_index] = classifier.predict_proba(X_test_cur)
        X_test_new[i] = classifier.predict_proba(X_test.as_matrix())
        i += 1
        mean_accur += accuracy_score(classifier.predict(X_test_cur), y_test_cur)
    mean_accur /= n_folds
    X_test_new_final = X_test_new.mean(axis=0)
    if test_mean == False:
        classifier.fit(X_train, y_train)
        new_x_test_final = classifier.predict_proba(X_test)
    return X_train_new, X_test_new_final, mean_accur

