# Authors: Martin Billinger <flkazemakase@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from nodeharvest import NodeHarvest


n = 100
plot_step = 0.02
solver = 'scipy_robust'     # 'cvx_robust' is faster, but requires cvxopt installed
np.random.seed(42)


def model(x, e=0.25):
    return np.prod(np.sin(2 * np.pi * x), axis=1) + np.random.randn(x.shape[0]) * e


x = np.random.rand(n, 2)
y = model(x)

rf = RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_leaf=3)
rf.fit(x, y)

nh = NodeHarvest(max_nodecount=None, solver=solver, verbose=True)
nh.fit(rf, x, y)

n_nodes = nh.coverage_matrix_.shape[1]
n_selected = np.sum(nh.get_weights() > 0)

xx, yy = np.meshgrid(np.arange(0, 1 + plot_step, plot_step), np.arange(0, 1 + plot_step, plot_step))
x_test = np.c_[xx.ravel(), yy.ravel()]

plt.figure()
z = rf.predict(x_test)
mse = np.var(z - model(x_test))
plt.contourf(xx, yy, z.reshape(xx.shape))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Random Forest: %d nodes, MSE=%f' % (n_nodes, mse))

plt.figure()
z = nh.predict(x_test)
mse = np.var(z - model(x_test))
plt.contourf(xx, yy, z.reshape(xx.shape))
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('Node Harvest: %d nodes, MSE=%f' % (n_selected, mse))


plt.show()
