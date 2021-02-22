# Support Vector Machines. You should implement (for sufficiently loose definitions of implement including
# "download") SVMs. This should be done in such a way that you can swap out kernel functions. I'd like to see at
# least two.


# KP Feb 9th at 11:14 PM
# for SVM are kernel functions the main hyper param to check ? Or should we look at more ?
# C. Higher C if you want to lower training scores, lower C otherwise ... if I recall correctly

import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from make_plots import plot_learning_curve
import model_complexity as mc
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVR
from sklearn.metrics import hamming_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import validation_curve as vc
from sklearn import svm
import json
import time
import timeit



def SVM(set1X_train, set1X_test, set1y_train, set1y_test,set2X_train, set2X_test, set2y_train, set2y_test):
    print("SVM")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    thisAlgFile = 'SVM1.txt'
    # Setting up the scaling pipeline
    pipeline_order = [('scaler', StandardScaler()), ('svm', svm.SVC(max_iter=1000, kernel='rbf'))]

    SVMpipe = Pipeline(pipeline_order)
    # Fitting the classfier to the scaled dataset
    svm_classifier_scaled1 = SVMpipe.fit(set1X_train, set1y_train)

    # Extracting the score
    print(svm_classifier_scaled1.score(set1X_train, set1y_train))
    # Testing accuracy on the test data
    svm_classifier_scaled1.score(set1X_test, set1y_test)
    print(svm_classifier_scaled1.score(set1X_test, set1y_test))

    # pipe2 = Pipeline(pipeline_order)

    # Fitting the classfier to the scaled dataset
    svm_classifier_scaled2 = SVMpipe.fit(set2X_train, set2y_train)

    # Extracting the score
    print(svm_classifier_scaled2.score(set2X_train, set2y_train))

    # Testing accuracy on the test data
    svm_classifier_scaled1.score(set2X_test, set2y_test)
    print(svm_classifier_scaled2.score(set2X_test, set2y_test))

    # Creating a grid of different hyperparameters
    grid_params = {
        'svm__C': np.logspace(-2, 10, 13),
        'svm__gamma': np.logspace(-9, 3, 13)
        #'svm__kernel': ['linear', 'rbf']#, 'poly', 'sigmoid']#, 'precomputed']
        #ccp_alpha': [0, .00005, .0002, .0003, .0004, .0005, .001]
        # 'svm__min_samples_leaf': [0.0001,0.001, 0.005,0.02,0.04, 0.06, 0.08, .1, .2, .5]
    }

    # Building a 10 fold Cross Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=svm_classifier_scaled1, param_grid=grid_params, scoring='accuracy', cv=3,
                               n_jobs=-1)

    # Fitting the grid to the training data
    grid_object.fit(set1X_train, set1y_train)

    # Extracting the best parameters
    print(grid_object.best_params_)

    rf_best1 = grid_object.best_estimator_

    print(rf_best1)

    # title = "Data1 SVM"
    # plt = plot_learning_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
    #                           train_sizes=np.linspace(.1, 1.0, 5))
    #
    # # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # plt.savefig('Data1 SVM LC'+timestr+'.png')
    # plt.show()

    # Building a 10 fold Cross Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=svm_classifier_scaled2, param_grid=grid_params, scoring='accuracy', cv=8,
                               n_jobs=-1)

    # # Fitting the grid to the training data
    # grid_object.fit(set2X_train, set2y_train)
    #
    # # Extracting the best parameters
    # print(grid_object.best_params_)
    # rf_best2 = grid_object.best_estimator_
    #
    # title = "Data2 SVM"
    # plt = plot_learning_curve(rf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
    #                           train_sizes=np.linspace(.1, 1.0, 5))
    #
    # # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # plt.savefig('Data2 SVM LC'+timestr+'.png')
    # plt.show()
    #
    # data1 = {'X_train': set1X_train, 'X_test': set1X_test, 'y_train': set1y_train, 'y_test': set1y_test}
    # data2 = {'X_train': set2X_train, 'X_test': set2X_test, 'y_train': set2y_train, 'y_test': set2y_test}
    #
    # # NEED TWO COMPLEXITY CURVES OF HYPERPARAMETERS:
    # # 1. Kernel
    # # 2. C
    #
    # Cs = [0.5,1,3,5,7,9,11,13]
    #     #np.logspace(-2, 10, 13)
    # vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "SVM", 'svm__C', Cs)
    #
    # vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "SVM", 'svm__C', Cs)
    #
    # pipeline_order = [('scaler', StandardScaler()),
    #                   ('svm', svm.SVC(max_iter=1000))]
    # SVMpipe_param1 = Pipeline(pipeline_order)
    #
    # pipeline_order = [('scaler', StandardScaler()),
    #                   ('svm', svm.SVC(max_iter=1000))]
    # SVMpipe_param2 = Pipeline(pipeline_order)

    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "SVM", 'svm__kernel',
    #                    kernels)
    #
    # vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "SVM", 'svm__kernel',
    #                    kernels)

    gammas= [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, .20, .25] #[0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.5, 2]
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "SVM", 'svm__gamma',
                       gammas)

    # vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "SVM", 'svm__gamma',
    #                    gammas)


    # configurations = [
    #     # {'estimator': DecisionTreeClassifier,
    #     #  'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf}, #, 'ccp_alpha': ccp_alpha},
    #     #  'changing_param': 'ccp_alpha',
    #     #  'changing_param_values': [0, .00005, .0002, .0003, .0004, .025],
    #     #  'complexity_label': 'ccp_alpha',
    #     #  'complexity_computer': lambda x: x.ccp_alpha,
    #     #  'prediction_performance_computer': accuracy_score,
    #     #  'prediction_performance_label': 'accuracy',
    #     #  'postfit_hook': lambda x: x,
    #     #  'data': data1,
    #     #  'n_samples': 30},
    #     {'estimator': DecisionTreeClassifier,
    #      'tuned_params': {'criterion': 'gini', 'random_state': 50,'ccp_alpha': ccp_alpha},  # , 'ccp_alpha': ccp_alpha},
    #      'changing_param': 'min_samples_leaf',
    #      'changing_param_values': [0.001, 0.005,0.02,0.04, 0.06, 0.08, .1, .2, .4, .5],
    #      'complexity_label': 'min_samples_leaf',
    #      'complexity_computer': lambda x: x.min_samples_leaf,
    #      'prediction_performance_computer': accuracy_score,
    #      'prediction_performance_label': 'accuracy',
    #      'postfit_hook': lambda x: x,
    #      'data': data1,
    #      'n_samples': 30},
    #     {'estimator': DecisionTreeClassifier,
    #      'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf},# , 'ccp_alpha': ccp_alpha},
    #      'changing_param': 'max_depth',
    #      'changing_param_values': [1,2,3,4,5,6,8,12,14,16],
    #      'complexity_label': 'max_depth',
    #      'complexity_computer': lambda x: x.max_depth,
    #      'prediction_performance_computer': accuracy_score,
    #      'prediction_performance_label': 'accuracy',
    #      'postfit_hook': lambda x: x,
    #      'data': data1,
    #      'n_samples': 30},
    # ]
    #
    # mc.plotMC(configurations)

    # best params after MC tuning:
    newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('svm',
                                   svm.SVC(max_iter=1500,
                                           C=3,
                                           gamma = .04,
                                           kernel='rbf'))])
    title = "SVM data1"
    plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data1 SVM Learning Curve' + timestr + '.png')
    plt.show()

    print("Final scores on train/test data Set1:")
    start = time.time()
    newrf_best1.fit(set1X_train, set1y_train)
    end = time.time()
    print('Train time: ', end - start)
    print('Train score: ', newrf_best1.score(set1X_train, set1y_train))
    start = time.time()
    print('Test score: ', newrf_best1.score(set1X_test, set1y_test))
    end = time.time()
    print('Test time: ', end - start)


    newrf_best2 = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('svm',
                                   svm.SVC(max_iter=1500,
                                                 C=1,
                                           gamma = 1,
                                                 kernel='rbf'))])
    title = "SVM data2"
    plt = plot_learning_curve(newrf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data2 SVM Learning Curve' + timestr + '.png')
    plt.show()

    print("Final scores on train/test data Set2:")
    start = time.time()
    newrf_best2.fit(set2X_train, set2y_train)
    end = time.time()
    print('Train time: ', end - start)
    print('Train score: ', newrf_best2.score(set2X_train, set2y_train))
    start = time.time()
    print('Test score: ', newrf_best2.score(set2X_test, set2y_test))
    end = time.time()
    print('Test time: ', end - start)
