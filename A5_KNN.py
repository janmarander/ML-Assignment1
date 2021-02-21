# k-Nearest Neighbors. You should "implement" (the quotes mean I don't mean it: steal the code) kNN. Use different
# values of k.

# don't forget to scale withKNN
# very, very important, practiaclly speaking
# make sure that all the dimensions have teh same scaling, or KNN will be dominated by the feature with the largest scale
# thats what standard scaler does i thought
#
#
# https://learning.oreilly.com/library/view/machine-learning-with/9781789343700/017bba40-8857-475f-b040-95ca4feeb509.xhtml
# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/

from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
import validation_curve as vc
import json
import time
import timeit

def KNN(set1X_train, set1X_test, set1y_train, set1y_test,set2X_train, set2X_test, set2y_train, set2y_test):
    print("KNN")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    thisAlgFile = 'KNN1.json'
    # Setting up the scaling pipeline
    pipeline_order = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(weights='uniform'))]
    pipe1 = Pipeline(pipeline_order)
    pipe1.fit(set1X_train, set1y_train)

    # Fitting the classfier to the scaled dataset
    knn_classifier_scaled1 = pipe1.fit(set1X_train, set1y_train)

    # Extracting the score
    print(knn_classifier_scaled1.score(set1X_train, set1y_train))
    # Testing accuracy on the test data
    knn_classifier_scaled1.score(set1X_test, set1y_test)
    print(knn_classifier_scaled1.score(set1X_test, set1y_test))

    # Fitting the classfier to the scaled dataset
    pipe2 = Pipeline(pipeline_order)
    knn_classifier_scaled2 = pipe2.fit(set2X_train, set2y_train)
    print(knn_classifier_scaled2.score(set2X_train, set2y_train))

    # Testing accuracy on the test data
    knn_classifier_scaled2.score(set2X_test, set2y_test)
    print(knn_classifier_scaled2.score(set2X_test, set2y_test))

    # # grid search
    # model = KNeighborsClassifier()
    # n_neighbors = range(1, 21, 2)
    # weights = ['uniform', 'distance']
    # metric = ['euclidean', 'manhattan', 'minkowski']
    # # define grid search
    # grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    # grid_result = grid_search.fit(X, y)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    # Initializing a grid with possible number of neighbors from 1 to 24
    grid = {'knn__n_neighbors': np.arange(1, 75),
            #'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'minkowski']}



    # Using cross validation to find optimal number of neighbors
    knn_grid1 = GridSearchCV(knn_classifier_scaled1, grid, cv=10)
    knn_grid1.fit(set1X_train, set1y_train)

    # Extracting the optimal number of neighbors
    print(knn_grid1.best_params_)
    # Extracting the accuracy score for optimal number of neighbors
    print(knn_grid1.best_score_)
    rf_best1 = knn_grid1.best_estimator_

    title = "KNN"
    plt = plot_learning_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    #fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plt.savefig('Data1 KNN LC'+timestr+'.png')
    plt.show()

    # Using cross validation to find optimal number of neighbors
    knn_grid2 = GridSearchCV(knn_classifier_scaled2, grid, cv=10)
    knn_grid2.fit(set2X_train, set2y_train)

    # Extracting the optimal number of neighbors
    print(knn_grid2.best_params_)
    # Extracting the accuracy score for optimal number of neighbors
    print(knn_grid2.best_score_)
    rf_best2 = knn_grid2.best_estimator_

    title = "KNN"
    plt = plot_learning_curve(rf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data2 KNN LC'+timestr+'.png')
    plt.show()

    # GENERATE MODEL COMPLEXITY CURVES!!!!

    data1 = {'X_train': set1X_train, 'X_test': set1X_test, 'y_train': set1y_train, 'y_test': set1y_test}

    # TUNED PARAMETERS:
    n_neighbors = 35
    ccp_alpha = (1e-1) * 10 ** -0.5
    max_depth = 6

    # NEED TWO COMPLEXITY CURVES OF HYPERPARAMETERS:
    # 1. N Neighbors (K)
    # 2. metric
    #metric in [‘euclidean’, ‘manhattan’, ‘minkowski’]
    #weights in [‘uniform’, ‘distance’]
    grid = {'knn__n_neighbors': np.arange(1, 50),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan', 'minkowski']}

    pipeline_order = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
    KNNpipe = Pipeline(pipeline_order)

    n_neighbors = np.arange(8, 100)
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "KNN", 'knn__n_neighbors', n_neighbors)

    vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "KNN", 'knn__n_neighbors', n_neighbors)

    pipeline_order = [('scaler', StandardScaler()),
                      ('knn', KNeighborsClassifier())]
    KNNpipe_param1 = Pipeline(pipeline_order)

    pipeline_order = [('scaler', StandardScaler()),
                      ('knn', KNeighborsClassifier())]
    KNNpipe_param2 = Pipeline(pipeline_order)

    metrics = ['euclidean', 'manhattan', 'minkowski']
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "KNN", 'knn__metric', metrics)

    vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "KNN", 'knn__metric', metrics)

    weights = ['uniform', 'distance']
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "KNN", 'knn__weights', weights)

    vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "KNN", 'knn__weights', weights)

    # configurations = [
    #     {'estimator': pipeMC,
    #      'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf},
    #      # , 'ccp_alpha': ccp_alpha},
    #      'changing_param': 'ccp_alpha',
    #      'changing_param_values': [0, .00005, .0002, .0003, .0004, .025],
    #      'complexity_label': 'ccp_alpha',
    #      'complexity_computer': lambda x: x.ccp_alpha,
    #      'prediction_performance_computer': accuracy_score,
    #      'prediction_performance_label': 'accuracy',
    #      'postfit_hook': lambda x: x,
    #      'data': data1,
    #      'n_samples': 30},
    #     {'estimator': pipeMC,
    #      'tuned_params': {'criterion': 'gini', 'random_state': 50, 'ccp_alpha': ccp_alpha},
    #      # , 'ccp_alpha': ccp_alpha},
    #      'changing_param': 'min_samples_leaf',
    #      'changing_param_values': [0.02, 0.04, 0.06, 0.08, .1, .2, .4, .5],
    #      'complexity_label': 'min_samples_leaf',
    #      'complexity_computer': lambda x: x.min_samples_leaf,
    #      'prediction_performance_computer': accuracy_score,
    #      'prediction_performance_label': 'accuracy',
    #      'postfit_hook': lambda x: x,
    #      'data': data1,
    #      'n_samples': 30},
    #     {'estimator': pipeMC,
    #      'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf},
    #      # , 'ccp_alpha': ccp_alpha},
    #      'changing_param': 'max_depth',
    #      'changing_param_values': [1, 2, 3, 4, 5, 6, 8, 12],
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
    #
    # data2 = {'X_train': set2X_train, 'X_test': set2X_test, 'y_train': set2y_train, 'y_test': set2y_test}
    # # TUNED PARAMETERS:
    # min_samples_leaf = 0.02
    # ccp_alpha = (1e-1) * 10 ** -0.5
    # max_depth = 6

    # best params after MC tuning:
    newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('knn',
                                   KNeighborsClassifier(n_neighbors=75,
                                           weights='uniform',
                                           metric='manhattan'))])
    title = "KNN data1"
    plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data1 KNN Learning Curve' + timestr + '.png')
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
                                  ('knn',
                                   KNeighborsClassifier(n_neighbors=9,
                                                        weights='uniform',
                                                        metric='euclidean'))])
    title = "KNN data2"
    plt = plot_learning_curve(newrf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data2 KNN Learning Curve' + timestr + '.png')
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


