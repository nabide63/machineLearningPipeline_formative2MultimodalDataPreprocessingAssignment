"""
numpy_ml.py
-----------
Pure-numpy ML classes (RandomForestClassifier, LabelEncoder).
Must be importable at model-load time since joblib pickle stores
the class path as 'numpy_ml.RandomForestClassifier'.
"""
import numpy as np


class LabelEncoder:
    def fit_transform(self, y):
        self.classes_ = np.array(sorted(set(y)))
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self._map[v] for v in y])

    def inverse_transform(self, y):
        return self.classes_[np.array(y)]


class RandomForestClassifier:
    """Minimal Random Forest using numpy decision stumps (no sklearn needed)."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []
        self.classes_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def _build(self, X, y, depth=0, max_depth=8):
        classes, counts = np.unique(y, return_counts=True)
        majority = classes[np.argmax(counts)]
        if len(classes) == 1 or depth >= max_depth or len(y) < 4:
            return majority
        n_feat = max(1, int(np.sqrt(X.shape[1])))
        feat_idx = self.rng.choice(X.shape[1], n_feat, replace=False)
        best_gain, best_feat, best_thr = -1, None, None
        parent_gini = self._gini(y)
        for f in feat_idx:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thrs = (vals[:-1] + vals[1:]) / 2
            if len(thrs) > 10:
                thrs = thrs[self.rng.choice(len(thrs), 10, replace=False)]
            for t in thrs:
                l, r = y[X[:, f] <= t], y[X[:, f] > t]
                if not len(l) or not len(r):
                    continue
                gain = parent_gini - (
                    len(l) / len(y) * self._gini(l) +
                    len(r) / len(y) * self._gini(r)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, t
        if best_feat is None:
            return majority
        m = X[:, best_feat] <= best_thr
        return {
            "f": best_feat, "t": best_thr,
            "l": self._build(X[m],  y[m],  depth + 1, max_depth),
            "r": self._build(X[~m], y[~m], depth + 1, max_depth),
        }

    def _pred1(self, tree, x):
        while isinstance(tree, dict):
            tree = tree["l"] if x[tree["f"]] <= tree["t"] else tree["r"]
        return tree

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(X)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, n, replace=True)
            self.trees_.append(self._build(X[idx], y[idx]))
        return self

    def predict(self, X):
        votes = np.array([[self._pred1(t, x) for t in self.trees_] for x in X])
        return np.array([
            np.bincount(r.astype(int), minlength=len(self.classes_)).argmax()
            for r in votes
        ])

    def predict_proba(self, X):
        votes = np.array([[self._pred1(t, x) for t in self.trees_] for x in X])
        return np.array([
            np.bincount(r.astype(int), minlength=len(self.classes_)) / len(r)
            for r in votes
        ])

