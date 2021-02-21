# Boosting. Implement or steal a boosted version of your decision trees. As before, you will want to use some
# form of pruning, but presumably because you're using boosting you can afford to be much more aggressive about
# your pruning.

from sklearn.ensemble import AdaBoostClassifier
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
from sklearn.pipeline import Pipeline
import validation_curve as vc
import json
import time
import timeit


def BDT(set1X_train, set1X_test, set1y_train, set1y_test,set2X_train, set2X_test, set2y_train, set2y_test):
    print("BDT")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    thisAlgFile = 'Boosting1.json'
    # Setting up the scaling pipeline
    pipeline_order = [('scaler', StandardScaler()), ('bdt', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200))]
    #NOTE USING MAX-DEPTH >2 WAS OVERFITTING

    BDTpipe = Pipeline(pipeline_order)
    # Fitting the classfier to the scaled dataset
    bdt_classifier_scaled1 = BDTpipe.fit(set1X_train, set1y_train)

    # Extracting the score
    print(bdt_classifier_scaled1.score(set1X_train, set1y_train))
    # Testing accuracy on the test data
    bdt_classifier_scaled1.score(set1X_test, set1y_test)
    print(bdt_classifier_scaled1.score(set1X_test, set1y_test))

    # pipe2 = Pipeline(pipeline_order)

    # Fitting the classfier to the scaled dataset
    bdt_classifier_scaled2 = BDTpipe.fit(set2X_train, set2y_train)

    # Extracting the score
    print(bdt_classifier_scaled2.score(set2X_train, set2y_train))
    # Testing accuracy on the test data
    bdt_classifier_scaled1.score(set2X_test, set2y_test)
    print(bdt_classifier_scaled2.score(set2X_test, set2y_test))

    # Creating a grid of different hyperparameters
    grid_params = {
        'bdt__n_estimators': [1000,2000,3000,4000,5000],
        'bdt__learning_rate':[(2**x)/100 for x in range(5)]+[1]
    }

    # Building a 10 fold Cross Validated GridSearchCV objertjgchjct
    grid_object = GridSearchCV(estimator=bdt_classifier_scaled1, param_grid=grid_params, scoring='accuracy', cv=8,
                               n_jobs=-1)

    # Fitting the grid to the training data
    grid_object.fit(set1X_train, set1y_train)

    # Extracting the best parameters
    print(grid_object.best_params_)
    rf_best1 = grid_object.best_estimator_

    print(rf_best1)

    title = "Data1 Boosted Decision Trees"
    plt = plot_learning_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plt.savefig('Data1 BDT LC'+timestr+'.png')
    plt.show()

    # Building a 10 fold Cross Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=bdt_classifier_scaled2, param_grid=grid_params, scoring='accuracy', cv=8,
                               n_jobs=-1)

    # Fitting the grid to the training data
    grid_object.fit(set2X_train, set2y_train)

    # Extracting the best parameters
    print(grid_object.best_params_)
    rf_best2 = grid_object.best_estimator_

    title = "Data2 Boosted Decision Trees"
    plt = plot_learning_curve(rf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))

    # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plt.savefig('Data2 BDT LC'+timestr+'.png')
    plt.show()

    data1 = {'X_train': set1X_train, 'X_test': set1X_test, 'y_train': set1y_train, 'y_test': set1y_test}
    data2 = {'X_train': set2X_train, 'X_test': set2X_test, 'y_train': set2y_train, 'y_test': set2y_test}

    # GENERATE MODEL COMPLEXITY CURVES!!!!

    # TUNED PARAMETERS:
    n_estimators = 100
    learning_rate = 1.28
    #max_depth = 14

    # NEED TWO COMPLEXITY CURVES OF HYPERPARAMETERS:
    # 1. Weak learners (n_estimators?)
    # 2. learning_rate
    n_estimators = [1000,1500,2000,2500,2800,3000,4000] #40,60,80,150,
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "Boosted Decision Tree", 'bdt__n_estimators', n_estimators)

    vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "Boosted Decision Tree", 'bdt__n_estimators', n_estimators)

    pipeline_order = [('scaler', StandardScaler()),
                      ('bdt', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200))]
    DTpipe_param1 = Pipeline(pipeline_order)

    pipeline_order = [('scaler', StandardScaler()),
                      ('bdt', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         algorithm="SAMME",
                         n_estimators=200))]
    DTpipe_param2 = Pipeline(pipeline_order)

    learning_rates = [.5,1,1.25,1.5,1.75,2.0] #.01,.02,.04,.06,
    vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "Boosted Decision Tree", 'bdt__learning_rate',
                       learning_rates)

    vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "Boosted Decision Tree", 'bdt__learning_rate',
                       learning_rates)

    # best params after MC tuning:
    newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('bdt',
                                   AdaBoostClassifier(n_estimators=1500,
                                                 learning_rate=1.5))])
    title = "Boosting data1"
    plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data1 Boosting Learning Curve' + timestr + '.png')
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
                                  ('bdt',
                                   AdaBoostClassifier(n_estimators=2000,
                                                 learning_rate=1.3))])
    title = "Boosting data2"
    plt = plot_learning_curve(newrf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                              train_sizes=np.linspace(.1, 1.0, 5))
    plt.savefig('Data2 Boosting Learning Curve' + timestr + '.png')
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
