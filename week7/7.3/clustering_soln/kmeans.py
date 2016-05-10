__author__ = 'jaredthompson'

import random
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

class KMeans:

    def __init__(self, k=1, max_iter=1000, init='random', dist=euclidean):

        # The following constant block is given to you.
        # Read it carefully to see which of these constants are being kept as local copies.
        # Reach goal: Implement kmeans++, so that if init='kmeans++' that initialization will be used.

        self.k = k
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.dist = dist
        self.init = init

    def fit(self, X):

        # Initialize the starting cluster centroids
        if self.init == 'random':
            # * initialize the cluster centers to be random points selected from X
            self.cluster_centers_ = [tuple(pt) for pt in random.sample(X, self.k)]
            
        # Optimize the clustering
        # * top of main loop: iterate until convergence you have hit max iterations
        for it in xrange(self.max_iter):

            # * you'll need to store the cluster assignments somehow. I recommend a defaultdict()
            clusters = defaultdict(list)

            # Assign each point to its closest centroid, forming a cluster
            # * for each datapoint in X
            for row in X:
                # * calculate the distance of that datapoint from each cluster center
                distance_from_centers = [self.dist(row, center) for center in self.cluster_centers_]
                # * store the index of the cluster center with the shortest distance to the data point
                center = tuple(self.cluster_centers_[np.argmin(distance_from_centers)])
                # * store the data point as belonging to that cluster center
                clusters[center].append(row)

            # Compute the new centroids based on the current cluster assignments
            # * you'll need to store the new centroid centers
            new_centers = []
            # * for each cluster
            for center, pts in clusters.iteritems():
                # * take the mean coordinates of all the points belonging to that cluster
                new_center = np.mean(pts, axis=0)
                # * store the new centroid
                new_centers.append(tuple(new_center))
                
            # * check the convergence criterion: if cluster centers do not differ from new_centers, stop
            if set(new_centers) == set(self.cluster_centers_):
                print it
                break

            # * you'll need to store the cluster centers to self.cluster_centers_
            self.cluster_centers_ = [tupl for tupl in new_centers]

    # I give you the predict function here.
    def predict(self, X):

        y_pred = []
        for row in X:
            distance_from_centers = [self.dist(row, center) for i, center in enumerate(self.cluster_centers_)]
            y_pred.append(np.argmin(distance_from_centers))

        return np.array(y_pred)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    n_samples = 500
    np.random.seed(42)
    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as sKMeans

    X,y = datasets.make_blobs(n_samples=500, cluster_std=1.0, random_state=3)
    X = StandardScaler().fit_transform(X)
    X.shape
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)
    plt.show()
    