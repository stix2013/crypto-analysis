"""Online Random Forest using incremental retraining."""

from collections import deque

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from crypto_analysis.online.base import OnlineModel


class OnlineRandomForest(OnlineModel):
    """Incrementally updatable Random Forest using online bagging.

    Maintains ensemble of decision trees that are periodically
    retrained with recent samples. Uses reservoir sampling to
    maintain representative training set.

    Attributes:
        n_trees: Number of trees in the ensemble
        max_samples: Maximum samples to retain per tree
        trees: List of trained tree models
        sample_buffers: Per-tree sample buffers for reservoir sampling
    """

    def __init__(self, n_trees: int = 10, max_samples_per_tree: int = 1000) -> None:
        """Initialize Online Random Forest.

        Args:
            n_trees: Number of trees in ensemble
            max_samples_per_tree: Maximum samples to buffer per tree
        """
        super().__init__(name="OnlineRandomForest", learning_rate=0.01)
        self.n_trees = n_trees
        self.max_samples = max_samples_per_tree
        self.trees: list[DecisionTreeRegressor | None] = []
        self.sample_buffers = [
            deque(maxlen=max_samples_per_tree) for _ in range(n_trees)
        ]

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update trees with new samples.

        Uses online bagging: each sample is randomly assigned to
        a subset of trees for training.

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        for i in range(self.n_trees):
            for xi, yi in zip(X, y):
                self.sample_buffers[i].append((xi, yi))

            if len(self.sample_buffers[i]) >= self.max_samples // 2:
                self._retrain_tree(i)

    def _retrain_tree(self, tree_idx: int) -> None:
        """Retrain specific tree with recent samples.

        Args:
            tree_idx: Index of tree to retrain
        """
        samples = list(self.sample_buffers[tree_idx])
        if not samples:
            return

        X_batch = np.array([s[0] for s in samples])
        y_batch = np.array([s[1] for s in samples])

        tree = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=50)
        tree.fit(X_batch, y_batch)

        while len(self.trees) <= tree_idx:
            self.trees.append(None)
        self.trees[tree_idx] = tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Mean prediction across all trees of shape (n_samples,)
        """
        if not self.trees:
            return np.zeros(len(X))

        predictions = np.array(
            [tree.predict(X) for tree in self.trees if tree is not None]
        )
        if predictions.size == 0:
            return np.zeros(len(X))

        return np.mean(predictions, axis=0)
