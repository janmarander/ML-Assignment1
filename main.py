# You should implement five learning algorithms. They are:
# • Decision trees with some form of pruning
# • Neural networks
# • Boosting
# • Support Vector Machines
# • k-nearest neighbors
#
# Each algorithm is described in detail in your textbook, the handouts, and all over the web. In fact,
# instead of implementing the algorithms yourself, you may (and by may I mean should) use software packages
# that you find elsewhere; however, if you do so you should provide proper attribution. Also, you will note
# that you have to do some fiddling to get good results, graphs and such, so even if you use another's package,
# you may need to be able to modify it in various ways.
#
# Decision Trees. For the decision tree, you should implement or steal a decision tree algorithm (and by
# "implement or steal" I mean "steal"). Be sure to use some form of pruning. You are not required to use
# information gain (for example, there is something called the GINI index that is sometimes used) to split
# attributes, but you should describe whatever it is that you do use.
#
# Neural Networks. For the neural network you should implement or steal your favorite kind of network and
# training algorithm. You may use networks of nodes with as many layers as you like and any activation function
# you see fit.
#
# Boosting. Implement or steal a boosted version of your decision trees. As before, you will want to use some
# form of pruning, but presumably because you're using boosting you can afford to be much more aggressive about
# your pruning.
#
# Support Vector Machines. You should implement (for sufficiently loose definitions of implement including
# "download") SVMs. This should be done in such a way that you can swap out kernel functions. I'd like to see at
# least two.
#
# k-Nearest Neighbors. You should "implement" (the quotes mean I don't mean it: steal the code) kNN. Use different
# values of k.
#
# Testing. In addition to implementing (wink) the algorithms described above, you should design two interesting
# classification problems. For the purposes of this assignment, a classification problem is just a set of training
# examples and a set of test examples. I don't care where you get the data. You can download some, take some from
# your own research, or make some up on your own. Be careful about the data you choose, though. You'll have to
# explain why they are interesting, use them in later assignments, and come to really care about them.

# WHAT TO TURN IN:
#•	the training and testing error rates you obtained running the various learning algorithms on your problems.
# At the very least you should include graphs that show performance on both training and test data as a function
# of training size (note that this implies that you need to design a classification problem that has more than a
# trivial amount of data) and--for the algorithms that are iterative--training times/iterations. Both of these
# kinds of graphs are referred to as learning curves, BTW.
#
# •	analyses of your results. Why did you get the results you did? Compare and contrast the different algorithms.
# What sort of changes might you make to each of those algorithms to improve performance? How fast were they in
# terms of wall clock time? Iterations? Would cross validation help (and if it would, why didn't you implement it?)?
# How much performance was due to the problems you chose? How about the values you choose for learning rates,
# stopping criteria, pruning methods, and so forth (and why doesn't your analysis show results for the different
# values you chose? Please do look at more than one. And please make sure you understand it, it only counts if the
# results are meaningful)? Which algorithm performed best? How do you define best? Be creative and think of as many
# questions you can, and as many answers as you can.

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

import make_plots
import A1_Decision_Trees as dt
import A5_KNN as knn
import A3_Boosting as bdt
import A4_SVM as svm
import A2_Neural_Network as nn
import sys
import time

if __name__ == '__main__':
    stdoutOrigin = sys.stdout
    print(stdoutOrigin)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    sys.stdout = open("log"+timestr+".txt", "w")
    print(timestr)

    print('ML Assignment 1')
    # read in data:
    set1 = pd.read_hdf('alldata.hdf', 'set1')
    ny = set1.shape[1] - 1
    set1X = set1.drop(ny, 1).copy().values
    set1Y = set1[ny].copy().values
    # print(set1)
    # print(set1Y)

    set2 = pd.read_hdf('alldata.hdf', 'set2b')
    ny = set2.shape[1] - 1
    set2X = set2.drop(ny, 1).copy().values
    set2Y = set2[ny].copy().values
    # print(set2)
    # print(set2Y)

    # split into train and test data
    set1X_train, set1X_test, set1y_train, set1y_test = sk.model_selection.train_test_split(set1X, set1Y, test_size=0.3,
                                                                                           random_state=42,
                                                                                           stratify=set1Y)
    set2X_train, set2X_test, set2y_train, set2y_test = sk.model_selection.train_test_split(set2X, set2Y, test_size=0.3,
                                                                                           random_state=42,

                                                                                           stratify=set2Y)

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("logDT"+timestr+".txt", "w")
    # #DT
    # dt.DT(set1X_train, set1X_test, set1y_train, set1y_test, set2X_train, set2X_test, set2y_train, set2y_test)

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("logKNN"+timestr+".txt", "w")
    #KNN
    knn.KNN(set1X_train, set1X_test, set1y_train, set1y_test, set2X_train, set2X_test, set2y_train, set2y_test)

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("logBDT"+timestr+".txt", "w")
    #BDT
    bdt.BDT(set1X_train, set1X_test, set1y_train, set1y_test, set2X_train, set2X_test, set2y_train, set2y_test)

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("logSVM"+timestr+".txt", "w")
    #SVM
    svm.SVM(set1X_train, set1X_test, set1y_train, set1y_test, set2X_train, set2X_test, set2y_train, set2y_test)

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("logNN"+timestr+".txt", "w")
    #NN
    nn.NN(set1X_train, set1X_test, set1y_train, set1y_test, set2X_train, set2X_test, set2y_train, set2y_test)

    sys.stdout.close()
    sys.stdout = stdoutOrigin