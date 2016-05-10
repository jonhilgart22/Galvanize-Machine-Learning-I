__author__ = 'jaredthompson'


import numpy as np
from scipy.spatial.distance import euclidean, cityblock

LARGE = 1.0e20

class KMedoids:
    '''
        Performs PAM clustering and fit. Stores initial clustering as self.clusters. Also provides a predict method.
    '''

    # I give you the constant block here
    def __init__(self, k=1, max_iter=10, dist=cityblock):
        self.k = k # k has the same meaning as it does in KMeans
        self.max_iter =  max_iter# This is a stopping condition only used in case of nonconvergence
        self.cluster_centers_ = None # These are centers but not medoids
        self.dist = dist
        self.medoids = None # this is the array of medoid assignments
        self.clusters_ = None # this is the array of cluster assignments
        self.D = None # This is the matrix of distances ( you do not have to use this )
        self.scores = [] # stores the scores of each cluster
        self.cost = LARGE
        
        
    # This function is optional. You do not have to use it, but you will need to compute distances somehow.
    def _compute_dist_matrix(self, X):
        '''
            X: m x n 2-d numpy array
            Returns a distance matrix.
        '''

        D = np.zeros((X.shape[0],X.shape[0]))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                D[i, j] = self.dist(xi, xj)
        return D


    @staticmethod
    def _configuration(X, D, clusters, medoids):
        '''
            X: m x n 2-d numpy array
            D: m x m 2-d numpy array
            clusters: 1-d numpy array of length m
            medoids: 1-d numpy array of length k

            Returns cost as a sum of distances to their respective medoids and performs clustering
        '''

        # * you'll want to store the cost of this configuration
        temp_cost = 0
        # * you'll need to store the temporary configuration somehow
        temp_clusters = np.zeros((len(clusters)), dtype=int)
        # * for each member of the data set
        for i in xrange(X.shape[0]):
            # * assign the data point the medoid closest to it. add this distance to the cost
            temp_clusters[i] = np.argmin([D[i, m] for m in medoids])
            temp_cost += D[i, medoids[temp_clusters[i]]]

        # * you'll want to return the cost for this configuration and the cluster assignments
        return temp_cost, temp_clusters



    # The following two functions are optional and you do not have to use them.
    @staticmethod
    def _swap_out(medoids, m, index):
        '''
            medoids: 1-d numpy array of length k
            m: int
            index: int

            Swaps medoid m in place with a chosen member "index" belonging to the dataset.
            Returns the swapped index and medoid
        '''

        swapped_index = index
        swapped_medoid = medoids[m]
        medoids[m] = swapped_index

        return swapped_index, swapped_medoid

    @staticmethod
    def _swap_back(medoids, m, swapped_medoid):
        '''
              medoids: 1-d numpy array of length k
              m: int
              swapped_medoid: int

              Swaps medoid m in place with the index number of the swapped medoid
        '''
        medoids[m] = swapped_medoid

    def fit(self, X):
        '''
            X: m x n 2-d numpy array

            Performs clustering and returns self.cluster_centers_
        '''
        # * precompute distances matrix
        self.D = self._compute_dist_matrix(X)

        # * randomly select k medoids from among the X
        self.medoids = np.random.choice(X.shape[0], self.k, replace=False)

        # * randomly assign medoids
        self.clusters = np.random.choice(self.medoids, X.shape[0], replace=True)

        # * mark the cluster centers
        self.cluster_centers_ = X[self.medoids]

        # Keep reconfiguring the medoids until a local minimum is found. This algorithm should converge quite rapidly.
        iter = 0 # * keep track of iterations and provide a stopping point (self.max_iter) to stop nonconverging loops.
        while iter < self.max_iter:

            print "iteration: %s" % iter

            # * you'll need to store the original medoids for comparison somehow
            old_medoids = list(self.medoids)

            # * following the pseudocode exactly: iterate over all medoids and nonmedoid points
            for m in xrange(self.medoids.shape[0]):
                for o in xrange(self.clusters.shape[0]):
                    if o != self.medoids[m]:

                        # * swap out each medoid with another point
                        swapped_index, swapped_medoid = self._swap_out(self.medoids, m, o)

                        # * calculate new cost (and clustering configuration)
                        new_cost, new_clusters = self._configuration(X, self.D, self.clusters, self.medoids)

                        # * if this configuration has a lower cost than the current cost
                        if (new_cost<self.cost):

                            # * store the new cost and clusters as the current clusters
                            self.cost = new_cost
                            self.clusters = new_clusters

                            # * you may also keep the current score for plotting
                            self.scores.append(self.cost)
                            
                        else:
                            # * if you actually swapped and recalculated the configuration in place
                            # * you'll need to swap it back
                            
                            self._swap_back(self.medoids, m, swapped_medoid)
            iter += 1
            # * if the medoids have stopped changing, stop optimization
            if set(self.medoids) == set(old_medoids):
                break

        # * at the end of the optimization, store the coordinates of the final medoids.
        self.cluster_centers_ = X[self.medoids]

    # I give this predict block to you
    def predict(self, X):
        '''
            X: m x n 2-d numpy array
        '''
        y_pred = []
        for i, row in enumerate(X): # this is just temporary
            y_pred.append(np.argmin([self.dist(row, center) for i, center in enumerate(self.cluster_centers_)]))
       
        return y_pred
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    n_samples = 130
    np.random.seed(42)
    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as sKMeans

    X,y = datasets.make_blobs(n_samples=n_samples, cluster_std=1.0, random_state=3)
    X = StandardScaler().fit_transform(X)
    X.shape
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    pam = KMedoids(k=3)
    pam.fit(X)  
    plt.scatter(X[:, 0], X[:, 1], color=colors[pam.clusters].tolist(), s=10)
    plt.show()
    plt.clf()
    plt.plot(pam.scores)
    plt.show()