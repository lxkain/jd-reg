import numpy as np
import matplotlib.pyplot as plt

from jdr import generate_spirals, GaussianMixtureRegressorAffineTransformation, GaussianMixtureRegressorMixingMeans

X_train, Y_train = generate_spirals()
X_test,  Y_test  = generate_spirals()

jdgmm = GaussianMixtureRegressorAffineTransformation(4, max_iter=1000, n_init=10)  # play with the number of components here, 1-32
#jdgmm = GaussianMixtureRegressorMixingMeans(48, max_iter=1000, n_init=10)  # play with the number of components here, 1-32
jdgmm.fit(X_train, Y_train)
Y_test_hat = jdgmm.predict(X_test)


n = 30  # grid density
xx, yy = np.meshgrid(np.linspace(-2, 2, n),
                     np.linspace(-2, 2, n))
X_grid = np.c_[xx.flat, yy.flat]
Y_grid_hat = jdgmm.predict(X_grid)


def draw_ellipses(gmm):
    from matplotlib.patches import Ellipse
    for k in range(gmm.n_components):
        t = gmm.covariance_type
        if t == 'full':
            c = gmm.covariances_[k][:2, :2]
        elif t == 'tied':
            c = gmm.covariances_[:2, :2]
        elif t == 'diag':
            c = np.diag(gmm.covariances_[k][:2])
        elif t == 'spherical':
            c = np.eye(gmm.means_.shape[1]) * gmm.covariances_[k]
        v, w = np.linalg.eigh(c)
        u = w[0] / np.linalg.norm(w[0])
        angle = 180 * np.arctan2(u[1], u[0]) / np.pi  # in degrees
        radius = 2 * np.sqrt(2. * v)   # 1 stdev = 68%
        xy = gmm.means_[k, :2]
        plt.text(xy[0], xy[1], k)
        ell = Ellipse(xy, radius[0], radius[1], 180 + angle, alpha=0.1, color='k')
        plt.gca().add_artist(ell)


def draw_grid(M, **kargs):
    def arrows(x, y, **kargs):
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, **kargs)
    for m in np.split(M, n):
        arrows(m[:,0], m[:,1], **kargs)


if 1:
    plt.figure()
    plt.plot(jdgmm.gmmX.predict_proba(X_test))

grid = False
plt.figure()
plt.subplot(211)
plt.plot(X_test[:,0], X_test[:,1], 'r.-', alpha=0.5)
draw_ellipses(jdgmm.gmmX)
if grid:
    draw_grid(X_grid, color='r', alpha=0.1)
plt.axis('equal')
plt.subplot(212)
plt.plot(Y_test[:,0], Y_test[:,1], 'g.-', alpha=0.5)
plt.plot(Y_test_hat[:,0], Y_test_hat[:,1], 'b.-', alpha=0.5)
draw_ellipses(jdgmm.gmmY)
if grid:
    draw_grid(Y_grid_hat, color='b', alpha=0.1)
plt.axis('equal');

plt.show()
