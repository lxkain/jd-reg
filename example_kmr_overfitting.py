import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from jdr import generate_spirals, KMeansRegressor


X_train, Y_train = generate_spirals(sigma=0.2)
X_test,   Y_test = generate_spirals(sigma=0.2)

K = np.arange(1, 256, 2)
E_train = np.empty(len(K))
E_test = np.empty(len(K))
for i, k in enumerate(K):
    print(k)
    kmr = KMeansRegressor(k)
    kmr.fit(X_train, Y_train)
    Y_train_hat = kmr.predict(X_train)
    Y_test_hat  = kmr.predict(X_test)
    E_train[i]  = mean_squared_error(Y_train, Y_train_hat)
    E_test[i]   = mean_squared_error(Y_test, Y_test_hat)

plt.semilogy(K, E_train, label='$E_{train}$')
plt.semilogy(K, E_test,  label='$E_{test}$')
plt.legend()
plt.xlabel('K')
plt.ylabel('E')
plt.show()

if 0:
    plt.subplot(211)
    plt.plot(X_test[:, 0], X_test[:, 1], 'r.-', label='$X_{test}$')
    for i, p in enumerate(kmr.X_cluster_centers):
        plt.text(p[0], p[1], i, ha='center', va='center')
    plt.axis('equal')
    plt.legend()
    plt.subplot(212)
    plt.plot(Y_test[:, 0], Y_test[:, 1], 'g.-', label='$Y_{test}$')
    for i, p in enumerate(kmr.Y_cluster_centers):
        plt.text(p[0], p[1], i, ha='center', va='center')
    plt.plot(Y_test_hat[:, 0], Y_test_hat[:, 1], 'b.-', alpha=0.5, label='$\hat{Y}_{test}$')
    plt.axis('equal')
    plt.legend()

    plt.show()
