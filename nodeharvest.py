# Authors: Martin Billinger <flkazemakase@gmail.com>
#
# Licence: BSD 3 clause

from collections import deque
import numpy as np
from scipy.linalg import svd
from sklearn.tree import _tree as ctree


class WeightedTree:
    def __init__(self, tree, w, m, s=[]):
        self.node_count = tree.node_count
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.feature = tree.feature
        self.threshold = tree.threshold
        self.weight = w
        self.value = m
        self.sample = s

    def predict(self, x):
        """Compute weighted sum of node means for all nodes each sample in x falls into."""
        n = x.shape[0]
        nr = np.arange(n)

        node = np.zeros(n, dtype=int)
        notleaf = np.ones(n, dtype=bool)

        y = np.zeros(n)
        w = np.zeros(n)

        while np.any(notleaf):
            w[notleaf] += self.weight[node[notleaf]]
            y[notleaf] += self.weight[node[notleaf]] * self.value[node[notleaf]]

            notleaf = self.children_left[node] != ctree.TREE_LEAF

            feature = self.feature[node]
            mask = x[nr, feature] <= self.threshold[node]
            left = np.flatnonzero(np.logical_and(mask, notleaf))
            right = np.flatnonzero(np.logical_and(~mask, notleaf))

            node[left] = self.children_left[node[left]]
            node[right] = self.children_right[node[right]]
        return y, w


class NodeHarvest:
    def __init__(self, max_nodecount=None, max_interaction=None, solver='scipy_robust', tolerance=1e-5,
                 **kwargs):
        self.w_root = None
        self.solver_args = kwargs
        self.max_nodecount = max_nodecount
        self.max_interaction = max_interaction
        self.solver = solver
        self.tolerance = tolerance
        self.coverage_matrix_ = np.zeros((0, 0))
        self.estimators_ = []

    def fit(self, forest, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        self.coverage_matrix_, tree_indices, node_indices = _compute_coverage_matrix(forest, x, self.max_nodecount,
                                                                                     self.max_interaction)

        means = np.dot(y, self.coverage_matrix_) / np.sum(self.coverage_matrix_, axis=0)

        solver = solvers[self.solver]
        w = solver(self.coverage_matrix_, y, means, **self.solver_args)
        w[np.abs(w) < self.tolerance] = 0

        self.estimators_ = []
        for nr, ti in enumerate(np.unique(tree_indices)):
            tree = forest.estimators_[ti].tree_
            ni = node_indices[tree_indices == ti]

            tree_weights = np.zeros(tree.node_count)
            tree_weights[ni] = w[tree_indices == ti]

            tree_means = np.zeros(tree.node_count)
            tree_means[ni] = means[tree_indices == ti]

            tree_samples = np.zeros(tree.node_count)
            tree_samples[ni] = np.sum(self.coverage_matrix_, 0)[tree_indices == ti]

            self.estimators_.append(WeightedTree(tree, tree_weights, tree_means, tree_samples))

    def predict(self, x):
        x = np.asarray(x)
        n = x.shape[0]

        y_total = np.zeros(n)
        w_total = np.zeros(n)
        for tree in self.estimators_:
            y, w = tree.predict(x)
            y_total += y
            w_total += w
        return y_total / w_total

    def get_weights(self):
        return np.hstack([tree.weight for tree in self.estimators_])

    def get_means(self):
        return np.hstack([tree.value for tree in self.estimators_])

    def get_samplecounts(self):
        return np.hstack([tree.sample for tree in self.estimators_])

    def get_featurecounts(self):
        return np.hstack([_tree_featurecount(tree) for tree in self.estimators_])


def _solve_cvx(i, y, node_means, nu=0.001, delta=1e-5, w0_min=0.001, maxiter=100, verbose=False):
    """Compute node weights with relaxed constraints using cvxopt."""
    from cvxopt.solvers import qp, options
    from cvxopt import matrix

    options['maxiters'] = maxiter
    options['show_progress'] = verbose

    n_samples, n_nodes = i.shape

    m = i * node_means

    p = np.dot(m.T, m) + np.eye(n_nodes) * nu
    p = (p.T + p) / 2
    q = -2 * np.dot(m.T, y)

    h1 = np.zeros((n_nodes, 1))
    h1[0] = -w0_min

    h2 = delta + np.ones((n_samples, 1))
    h3 = delta - np.ones((n_samples, 1))

    g = np.vstack([-np.eye(n_nodes), i, -i])
    h = np.vstack([h1, h2, h3])

    result = qp(matrix(p), matrix(q), matrix(g), matrix(h))
    w = np.ravel(result['x'])
    return w


def _solve_cvx2(i, y, node_means, nu=0.001, w0_min=0.001, maxiter=100, verbose=False):
    """Compute node weights with strict constraints using cvxopt."""
    from cvxopt.solvers import qp, options
    from cvxopt import matrix

    options['maxiters'] = maxiter
    options['show_progress'] = verbose

    n_samples, n_nodes = i.shape

    m = i * node_means
    b = _null_space(i.T)

    n_dims = b.shape[1]

    mb = np.dot(m, b)
    p = np.dot(mb.T, mb) + np.eye(n_dims) * nu * 2
    p = (p.T + p) / 2
    q = 2 * nu * b[0, :].T - 2 * np.dot(mb.T, y - np.mean(y))
    g = -b
    h = np.zeros((n_nodes, 1))
    h[0] = 1 - w0_min

    result = qp(matrix(p), matrix(q), matrix(g), matrix(h))

    d = np.ravel(result['x'])
    w = np.dot(b, d)
    w[0] += 1
    return w


def _solve_scipy(i, y, node_means, nu=0.001, delta=1e-5, w0_min=0.001, maxiter=100, verbose=False):
    """Compute node weights with relaxed constraints using scipy."""
    from scipy.optimize import minimize

    m = i * node_means
    mm = np.dot(m.T, m) + np.eye(m.shape[1]) * nu
    mm = (mm.T + mm) / 2

    def objective(w, y=y, m=m, mm=mm):
        wmm = np.dot(w.T, mm)
        ym2 = np.dot(y.T, m) * 2
        fun = np.dot(wmm, w) - np.dot(ym2, w)
        jac = 2 * wmm - ym2
        return fun, jac

    def constraint(w, i=i, delta=delta, w0_min=w0_min):
        iw = np.dot(i, w)
        c2 = delta + 1 - iw
        c3 = delta - 1 + iw
        c = np.hstack([w, c2, c3])
        c[0] -= w0_min
        return c

    w0 = np.zeros(i.shape[1])
    w0[0] = 1

    result = minimize(objective, w0, jac=True, options={'disp': verbose, 'maxiter': maxiter},
                      constraints={'type': 'ineq', 'fun': constraint})
    w = result['x']
    return w


def _solve_scipy2(i, y, node_means, nu=0.001, w0_min=0.001, maxiter=100, verbose=False):
    """Compute node weights with strict constraints using scipy."""
    from scipy.optimize import minimize

    m = i * node_means
    b = _null_space(i.T)
    mb = np.dot(m, b)

    n_dims = b.shape[1]

    def objective(d, nu=nu, mb=mb, b=b, mby=np.dot(mb.T, y - np.mean(y))):
        mbd = np.dot(mb, d)
        ridge = np.dot(b, d)
        ridge[0] += 1
        fun = np.dot(mbd.T, mbd) - 2 * d.dot(mby) + np.sum(ridge**2) * nu
        jac = 2 * (np.dot(mb.T, mb) + np.eye(n_dims) * nu).dot(d) - 2 * (mby - nu * b[0, :].T)
        return fun, jac

    def constraint(d):
        c = np.dot(b, d)
        c[0] += 1 - w0_min
        return c

    d0 = np.zeros(n_dims)
    result = minimize(objective, d0, jac=True, options={'disp': verbose, 'maxiter': maxiter},
                      constraints={'type': 'ineq', 'fun': constraint})

    d = result['x']
    w = np.dot(b, d)
    w[0] += 1
    return w


solvers = {'cvx_robust': _solve_cvx,
           'cvx_fast': _solve_cvx2,
           'scipy_robust': _solve_scipy,
           'scipy_fast': _solve_scipy2}


def _forest_apply_generator(forest, x, interactions=False):
    """Generator that computes coverage for all nodes in a forest."""
    for nt, tree in enumerate(forest.estimators_):
        i = _tree_apply(tree.tree_, x)
        if interactions:
            tfc = _tree_featurecount(tree.tree_)
            for ni, (column, nf) in enumerate(zip(i.T, tfc)):
                yield tuple(column), nt, ni, nf
        else:
            for ni, column in enumerate(i.T):
                yield tuple(column), nt, ni, None


def _compute_coverage_matrix(forest, x, max_nodecount, max_interaction):
    """Compute coverage matrix I.
    """

    nodes = {}
    for (i, nt, ni, nf) in _forest_apply_generator(forest, x, max_interaction):
        # limit feature interaction
        if max_interaction and nf > max_interaction:
            continue

        # don't include multiple nodes that contain the same observations
        if i in nodes:
            continue

        nodes[i] = (nt, ni)

        # limit maximum number of nodes
        if max_nodecount and len(nodes) >= max_nodecount:
            break

    i = sorted(nodes.keys())[::-1]
    indices = np.asarray([nodes[k] for k in i])
    tree_indices = indices[:, 0]
    node_indices = indices[:, 1]
    i = np.asarray(i).T

    return i, tree_indices, node_indices


def _tree_featurecount(tree):
    """Count number of features that contribute to each node in the tree"""
    features = [set() for _ in range(tree.node_count)]
    queue = deque([0])
    while queue:
        i = queue.pop()
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != ctree.TREE_LEAF:
            features[l].update(features[i], {tree.feature[i]})
            features[r].update(features[i], {tree.feature[i]})
            queue.extend([l, r])
    nf = [len(f) for f in features]
    return nf


def _tree_bounds(tree, n_features=None):
    """Compute final decision rule for each node in tree"""
    if n_features is None:
        n_features = np.max(tree.feature) + 1
    aabbs = [AABB(n_features) for _ in range(tree.node_count)]
    queue = deque([0])
    while queue:
        i = queue.pop()
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != ctree.TREE_LEAF:
            aabbs[l], aabbs[r] = aabbs[i].split(tree.feature[i], tree.threshold[i])
            queue.extend([l, r])
    return aabbs


def _tree_apply(tree, x):
    """Compute coverage matrix of a single tree.

    This function behaves like sklearn.tree._tree.Tree.apply() except for the following differences:
        1. It finds all nodes (leafs *and* internal nodes) for each sample in `x`.
        2. Instead of a list of node indices it returns a boolean mask.
    """
    n = x.shape[0]
    nr = np.arange(n)
    i = np.zeros((n, tree.node_count))

    node = np.zeros(n, dtype=int)
    leaf = np.zeros(n, dtype=bool)

    while not np.all(leaf):
        i[nr, node] = True

        leaf = tree.children_left[node] == ctree.TREE_LEAF

        feature = tree.feature[node]
        mask = x[nr, feature] <= tree.threshold[node]
        left = np.flatnonzero(np.logical_and(mask, ~leaf))
        right = np.flatnonzero(np.logical_and(~mask, ~leaf))

        node[left] = tree.children_left[node[left]]
        node[right] = tree.children_right[node[right]]

    return i


def _null_space(a, eps=1e-15):
    """Compute null space of matrix A"""
    u, s, vh = svd(a, True)
    r = np.sum(s > eps)
    null_space = u[:, r:]
    return null_space


def _unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a, uidx = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=True)
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1])), uidx


class AABB:
    def __init__(self, n_features):
        self.limits = np.array([[-np.inf, np.inf]] * n_features)

    def split(self, f, v):
        left = AABB(self.limits.shape[0])
        right = AABB(self.limits.shape[0])
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()

        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right

    def rules(self, feature_names=None):
        if feature_names is None:
            feature_names = ['x_%d' % i for i in range(self.limits.shape[0])]
        rulestrings = []
        for n, (l, h) in zip(feature_names, self.limits):
            if (l, h) == (-np.inf, np.inf):
                continue
            string = n
            if l > -np.inf:
                string = '%.2f' % l + ' < ' + string
            if h < np.inf:
                string = string + ' <= ' + '%.2f' % h
            rulestrings.append(string)
        return rulestrings


#----------------------------------------------------------------------------------------------------------------------
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_array_less
from numpy.testing.utils import assert_array_compare


def assert_array_lesseq(x, y, tol=1e-8):
    assert np.all([a <= b + tol for a, b in np.broadcast(x, y)])


def assert_weight_constraints(i, w, delta=1e-5, w0=0.001):
    # make sure all weights are positive and the weight of the root node is at least w0
    assert_array_lesseq([w0] + [0] * (len(w) - 1), w)

    # make sure all weights sum to 1 +/- delta for each (training) sample
    assert_array_lesseq(np.dot(i, w), 1 + delta)
    assert_array_lesseq(1 - delta, np.dot(i, w))


def generate_dummy_forest():
    class DummyTree:
        def __init__(self, l, r, f, t, v):
            self.node_count = len(l)
            self.children_left = np.asarray(l)
            self.children_right = np.asarray(r)
            self.feature = np.asarray(f)
            self.threshold = np.asarray(t)
            self.value = np.asarray(v)

    # Tree 1 node indices
    # +-------+-------+   +-------+-------+   +---------------+
    # |       |       |   |       |       |   |               |
    # |   4   |   6   |   |       |       |   |               |
    # |       |       |   |       |       |   |               |
    # +-------+-------+   |   1   |   2   |   |       0       |
    # |       |       |   |       |       |   |               |
    # |   3   |   5   |   |       |       |   |               |
    # |       |       |   |       |       |   |               |
    # +-------+-------+   +-------+-------+   +---------------+
    tree1 = DummyTree(l=[1, 3, 5, -1, -1, -1, -1],
                      r=[2, 4, 6, -1, -1, -1, -1],
                      f=[0, 1, 1, -2, -2, -2, -2],
                      t=[0, 0, 0, 0, 0, 0, 0],
                      v=[0, 0, 0, -2, 2, 2, -2])

    # Tree 2 node indices
    # +---+-----------+   +---------------+   +---------------+   +---------------+   +---------------+
    # | 3 |     4     |   |       1       |   |       1       |   |       1       |   |               |
    # +---+---+---+---+   +-------+---+---+   +-------+-------+   +---------------+   |               |
    # |   8   | 12|   |   |       |   |   |   |       |       |   |               |   |               |
    # +---+---+---+   |   |       |   |   |   |       |       |   |               |   |       0       |
    # |       |   | 10|   |   5   | 9 | 10|   |   5   |   6   |   |       2       |   |               |
    # |   7   | 11|   |   |       |   |   |   |       |       |   |               |   |               |
    # |       |   |   |   |       |   |   |   |       |       |   |               |   |               |
    # +---+---+---+---+   +---+---+---+---+   +-------+-------+   +---------------+   +---------------+
    tree2 = DummyTree(l=[2,  3, 5, -1, -1, 7,  9, -1, -1, 11, -1, -1, -1],
                      r=[1,  4, 6, -1, -1, 8, 10, -1, -1, 12, -1, -1, -1],
                      f=[1,  0, 0, -2, -2, 1,  0, -2, -2,  1, -2, -2, -2],
                      t=[2, -2, 0,  0,  0, 0,  2,  0,  0,  0,  0,  0,  0],
                      v=[0,  0,  0, 1, -1, -1.0/3.0, 1.0/3.0, -2, 3, 0, 1, 3, -3])

    class DummyTreeWrapper:
        def __init__(self, t):
            self.tree_ = t

    class DummyForest:
        estimators_ = [DummyTreeWrapper(t) for t in [tree1, tree2]]

        def fit(self, x, y):
            return self

    forest = DummyForest()

    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [-3, -3], [-3, 3], [3, -3], [3, 3]])
    y = np.array([-3, 3, 3, -3, -1, 1, 1, -1])
    y_ = np.array([-2, 3, 3, -3, -2, 1, 1, -1])

    i1 = [[1, 1, 0, 1, 0, 0, 0],
          [1, 1, 0, 0, 1, 0, 0],
          [1, 0, 1, 0, 0, 1, 0],
          [1, 0, 1, 0, 0, 0, 1]]
    i1 = np.vstack([i1, i1])

    i2 = [[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
          [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
          [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
          [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

    return forest, tree1, tree2, i1, i2, x, y, y_


def test_nodeharvest():
    forest, tree1, tree2, i1, i2, x, y, y_ = generate_dummy_forest()

    nh = NodeHarvest(w0_min=0)
    nh.fit(forest, x, y)

    assert_array_almost_equal(nh.predict(x), y_, decimal=2)
    assert_array_almost_equal(nh.estimators_[0].weight, [0, 0, 0, 1, 0, 0, 0], decimal=2)
    assert_array_almost_equal(nh.estimators_[1].weight, [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1], decimal=2)


def test_forest_interface():
    forest, tree1, tree2, i1, i2, x, y, y_ = generate_dummy_forest()

    i = _tree_apply(tree1, x)
    assert_array_equal(i, i1)

    i = _tree_apply(tree2, x)
    assert_array_equal(i, i2)

    i, ti, ni = _compute_coverage_matrix(forest, x, max_nodecount=np.inf, max_interaction=np.inf)
    i_ref, _ = _unique_rows(np.hstack([i1, i2]).T)
    i_ref = i_ref[::-1].T
    assert_array_equal(i, i_ref)

    assert_array_equal(np.unique(ti), [0, 1])  # two trees
    assert_array_less(-1, ni)
    assert_array_less(ni[ti == 0], tree1.node_count)
    assert_array_less(ni[ti == 1], tree2.node_count)

    for i, f in enumerate(_tree_featurecount(tree1)):
        assert f <= min(2, i)

    for i, f in enumerate(_tree_featurecount(tree2)):
        assert f <= min(2, i)


def test_solvers():
    i = np.array([[1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [1, 0, 0, 0, 1]])

    m = 42 + np.array([0, -1, 1, 1, -1])
    y = 42 + np.array([[-1, 1, 1, -1]], dtype='float').T

    nu = 0.001
    delta = 1e-10
    w0_min = 0.001

    w_expected = [w0_min, 1 - w0_min, 1 - w0_min, 1 - w0_min, 1 - w0_min]

    w = _solve_scipy(i, y, m, nu, delta, w0_min)
    assert_weight_constraints(i, w, delta, w0_min)
    assert_almost_equal(w, w_expected, decimal=5)

    w = _solve_scipy2(i, y, m, nu, w0_min)
    assert_weight_constraints(i, w, delta, w0_min)
    assert_almost_equal(w, w_expected, decimal=5)

    w = _solve_cvx(i, y, m, nu, delta, w0_min)
    assert_weight_constraints(i, w, delta, w0_min)
    assert_almost_equal(w, w_expected, decimal=5)

    w = _solve_cvx2(i, y, m, nu, w0_min)
    assert_weight_constraints(i, w, delta, w0_min)
    assert_almost_equal(w, w_expected, decimal=5)


def test_utils():
    a = np.array([[0]])
    n = _null_space(a)
    assert_array_equal(n.shape, [1, 1])
    assert n != 0

    a = np.array([[2, 1], [1, 2]])
    n = _null_space(a)
    assert_array_equal(n.shape, [2, 0])

    a = np.array([[2, 1], [-4, -2]])
    n = _null_space(a)
    assert_array_equal(n.shape, [2, 1])
    assert_almost_equal(n[0] / n[1], 2.0)

    a = np.eye(42)
    u, i = _unique_rows(a)
    assert_array_equal(u[sorted(i)[::-1], :], a)
    assert_array_equal(u, a[i, :])

    a = np.ones((42, 42))
    u, i = _unique_rows(a)
    assert_array_equal(u, np.ones((1, 42)))
    assert_array_equal(u, a[i, :])


if __name__ == '__main__':
    test_utils()
    test_solvers()
    test_forest_interface()
    test_nodeharvest()
