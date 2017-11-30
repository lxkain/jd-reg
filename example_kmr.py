import matplotlib.pyplot as plt

from jdr import generate_spirals, KMeansRegressor


X_train, Y_train = generate_spirals()
X_test,   Y_test = generate_spirals()

kmr = KMeansRegressor(36)
kmr.fit(X_train, Y_train)
Y_test_hat = kmr.predict(X_test)

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
