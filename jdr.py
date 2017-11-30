import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def generate_spirals(N: int=1000, sigma: float=0.1) -> (np.ndarray, np.ndarray):
    t = np.linspace(0, 2.0, N) * 2 * np.pi
    r = np.linspace(2, .1, N)
    X = np.c_[-r * np.sin(t), r * np.cos(t)]
    r = np.linspace(0.9, .4, N)
    Y = np.c_[r * np.cos(t), r * np.sin(t)]
    X += np.random.rand(N, 2) * sigma  # add noise
    Y += np.random.rand(N, 2) * sigma  # add noise
    return X, Y


class KMeansRegressor(object):
    """
    Joint-Density K-Means Model for regression
    """
    def __init__(self, K: int):
        self.kmeans = KMeans(n_clusters=K)
        self.X_cluster_centers = None
        self.Y_cluster_centers = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Z = np.c_[X, Y]
        self.kmeans.fit(Z)
        d = X.shape[1]  # dimensionality
        self.X_cluster_centers = self.kmeans.cluster_centers_[:, :d]
        self.Y_cluster_centers = self.kmeans.cluster_centers_[:, d:]
        self.kmeans.cluster_centers_ = self.X_cluster_centers  # TODO: is there a better way?

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.Y_cluster_centers[self.kmeans.predict(X)]

# TODO: KMedoidRegressor


class KMeansRegressorDiff(KMeansRegressor):
    """
    This assumes that the local covariances are very similar between X and Y
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        index = self.kmeans.predict(X)
        return self.Y_cluster_centers[index] + X - self.X_cluster_centers[index]


class GaussianMixtureRegressorMixingMeans(object):
    """
    Joint-Density Gaussian Mixture Model for regression
    """
    def __init__(self, Q: int, **kwargs):
        self.xd = None
        self.yd = None
        self.gmmZ = GaussianMixture(n_components=Q, **kwargs)
        self.gmmX = None
        self.gmmY = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        gmmZ = self.gmmZ
        Q = self.gmmZ.n_components
        self.xd = xd = X.shape[1]  # dimensionality of features
        self.yd = yd = Y.shape[1]
        # create and model the joint distribution
        Z = np.hstack((X, Y))
        gmmZ.fit(Z)
        # manually create the associated GMM for X
        self.gmmX = gmmX = GaussianMixture(n_components=Q, max_iter=1)
        gmmX.fit(X)  # really just to create covariance objects, which we will then override
        gmmX.weights_ = gmmZ.weights_
        gmmX.means_ = gmmZ.means_[:, :xd]
        gmmX.covariances_ = gmmZ.covariances_[:, :xd, :xd]
        # manually create the associated GMM for Y
        self.gmmY = gmmY = GaussianMixture(n_components=Q, max_iter=1)
        gmmY.fit(Y)  # really just to create covariance objects, which we will then override
        gmmY.weights_ = gmmZ.weights_
        gmmY.means_ = gmmZ.means_[:, xd:]
        gmmY.covariances_ = gmmZ.covariances_[:, xd:, xd:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((len(X), self.yd))
        H = self.gmmX.predict_proba(X)
        for q in range(self.gmmX.n_components):
            Y += np.outer(H[:, q], self.gmmY.means_[q])  # probabilistically mixing together linear transforms
        return Y


class GaussianMixtureRegressorAffineTransformation(object):
    """
    Joint-Density Gaussian Mixture Model for regression
    """
    def __init__(self, Q: int, **kwargs):
        self.xd = None
        self.yd = None
        self.gmmZ = GaussianMixture(n_components=Q, **kwargs)
        self.gmmX = None
        self.gmmY = None
        self.W = None
        self.b = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        gmmZ = self.gmmZ
        Q = self.gmmZ.n_components
        self.xd = xd = X.shape[1]  # dimensionality of features
        self.yd = yd = Y.shape[1]
        # create and model the joint distribution
        Z = np.hstack((X, Y))
        gmmZ.fit(Z)
        # manually create the associated GMM for X
        self.gmmX = gmmX = GaussianMixture(n_components=Q, max_iter=1)
        with warnings.catch_warnings():  # TODO: is there a better way?
            warnings.simplefilter("ignore")
            gmmX.fit(X)  # really just to create covariance objects, which we will then override
        gmmX.weights_ = gmmZ.weights_
        gmmX.means_ = gmmZ.means_[:, :xd]
        gmmX.covariances_ = gmmZ.covariances_[:, :xd, :xd]
        # manually create gmmY,  for visualization or debug purposes
        self.gmmY = gmmY = GaussianMixture(n_components=Q, max_iter=1)
        with warnings.catch_warnings():  # TODO: is there a better way?
            warnings.simplefilter("ignore")
            gmmY.fit(Y)  # really just to create covariance objects, which we will then override
        gmmY.weights_ = gmmZ.weights_
        gmmY.means_ = gmmZ.means_[:, xd:]
        gmmY.covariances_ = gmmZ.covariances_[:, xd:, xd:]
        # regression
        self.W = np.empty((Q, xd, yd))
        self.b = np.empty((Q, yd))
        for q in range(Q):
            Sigma = gmmZ.covariances_[q]
            self.W[q] = W = np.linalg.solve(Sigma[:xd, :xd], Sigma[xd:, :xd].T)
            self.b[q] = gmmZ.means_[q, xd:] - gmmZ.means_[q, :xd] @ W

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((len(X), self.yd))
        H = self.gmmX.predict_proba(X)
        for q in range(self.gmmX.n_components):
            Y += (H[:, q] * (X @ self.W[q] + self.b[q]).T).T  # probabilistically mixing together linear transforms
        return Y
