# Authors: Martin Billinger <flkazemakase@gmail.com>
#
# Licence: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from nodeharvest import NodeHarvest, _tree_bounds

from sklearn.datasets import load_boston

np.random.seed(12345)

solver = 'cvx_robust'     # 'scipy_robust' does not converge well

boston = load_boston()

n = boston.data.shape[0]

training = np.arange(0, n, 2)
testing = np.arange(1, n, 2)

rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=20)
rf.fit(boston.data[training, :], boston.target[training])

nh = NodeHarvest(max_nodecount=500, max_interaction=2, solver=solver, verbose=True, tolerance=1e-2)
nh.fit(rf, boston.data[training, :], boston.target[training])

n_nodes = nh.coverage_matrix_.shape[1]
n_selected = np.sum(nh.get_weights() > 0)

print(n_selected, '/', n_nodes)


weights = nh.get_weights()
means = nh.get_means()
samples = nh.get_samplecounts()
nfeatures =nh.get_featurecounts()

rules = []
for tree in nh.estimators_:
    for node in _tree_bounds(tree):
        rules.append('\n'.join(node.rules(boston.feature_names)))

r0 = np.max(np.sqrt(weights[weights > 0]))

color = ['k', 'g', 'c', 'b', 'm', 'r', 'r', 'r', 'r', 'r']

for w, m, s, f, r in zip(weights, means, samples, nfeatures, rules):
    if w > 0:
        plt.plot(m, s, color[f] + '.', markersize=100 * np.sqrt(w) / r0, alpha=0.5)
        plt.text(m, s, r, va='center', ha='center')

plt.xlabel("Median value of owner-occupied homes in $1000's")
plt.ylabel("Number of samples in node")

plt.tight_layout()
plt.show()
