# create synthetic datasets with different properties
# https://github.com/faizanahemad/data-science/blob/master/exploration_projects/imbalance-noise-oversampling/lib.py

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from IPython.display import display
import time

def visualize_3d(X, y, algorithm="tsne", title="Data in 3D"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Xorig=X

    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    if X.shape[1] > 3:
        X = reducer.fit_transform(X)
    else:
        if type(X) == pd.DataFrame:
            X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 0],
            y=X1[:, 1],
            z=X1[:, 2],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    plot(fig)

def visualize_3d1(X, y, algorithm="tsne", title="Data in 3D"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    # if X.shape[1] > 3:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X) == pd.DataFrame:
    #         X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 2],
            y=X1[:, 3],
            z=X1[:, 4],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    #fig.savefig('Data' + timestr + '.png')
    plot(fig)

def visualize_3d2(X, y, algorithm="tsne", title="Data in 3D"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    # if X.shape[1] > 3:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X) == pd.DataFrame:
    #         X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 6],
            y=X1[:, 7],
            z=X1[:, 8],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 7'),
            yaxis=dict(
                title='Dim 8'),
            zaxis=dict(
                title='Dim 9'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    #fig.savefig('Data' + timestr + '.png')
    plot(fig)

def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    # if X.shape[1]>2:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X)==pd.DataFrame:
    #     	X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title(title);
    plt.show();

# 9 features, 5 useful, 4 not, 2 clusters per class
X1,y1 = make_classification(n_samples=10000, n_features=9, n_informative=5, n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1,class_sep=0.5,flip_y=0, random_state=17)

df1 = pd.DataFrame(np.c_[X1, y1])
print(df1.shape)
#df1.to_hdf('alldata.hdf','set1',complib='blosc',complevel=9)

# # 9 features, 5 useful, 3 redundant, 1 repeated, 1 cluster per class
# X2a,y2a = make_classification(n_samples=10000, n_features=9, n_informative=5, n_redundant=3, n_repeated=1, n_classes=2,
#                           n_clusters_per_class=1,class_sep=0.5,flip_y=0, random_state=17)
#
# df2a = pd.DataFrame(np.c_[X2a, y2a])
# print(df2a.shape)
# df2a.to_hdf('alldata.hdf','set2a',complib='blosc',complevel=9)

# 9 features, 5 useful, 3 redundant, 1 repeated, 2 clusters per class
X2b,y2b = make_classification(n_samples=10000, n_features=9, n_informative=5, n_redundant=3, n_repeated=1, n_classes=2,
                          n_clusters_per_class=2,class_sep=0.5,flip_y=0, random_state=17)


df2b = pd.DataFrame(np.c_[X2b, y2b])
print(df2b.shape)
print(df2b)
#df2b.to_hdf('alldata.hdf','set2b',complib='blosc',complevel=9)

#visualize_3d(X1,y1,algorithm="pca")
# visualize_3d(X2a,y2a,algorithm="pca")
visualize_3d(X2b,y2b,algorithm="pca")

#visualize_3d1(X1,y1,algorithm="pca")
# visualize_3d(X2a,y2a,algorithm="pca")
#visualize_3d1(X2b,y2b,algorithm="pca")

# visualize_3d2(X1,y1,algorithm="pca")
# visualize_3d2(X2a,y2a,algorithm="pca")
# visualize_3d2(X2b,y2b,algorithm="pca")