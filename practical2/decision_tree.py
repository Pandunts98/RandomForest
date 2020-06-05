import numpy as np


class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 false_branch=None,
                 true_branch=None,
                 is_leaf: bool = False):
        self.data = data
        self.labels = labels
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf


class DecisionTree:

    def __init__(self,
                 max_tree_depth=4,
                 criterion="gini"):
        self.tree = None
        self.max_depth = max_tree_depth

        if criterion == "entropy":
            self.criterion = self._entropy
        elif criterion == "gini":
            self.criterion = self._gini
        else:
            raise RuntimeError(f"Unknown criterion: '{criterion}'")

    @staticmethod
    def _gini(labels: np.ndarray) -> float:

        classes, counts = np.unique(labels, return_counts=True)
        length = len(labels)
        return 1 - sum([(i / length) ** 2 for i in counts])

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:

        classes, counts = np.unique(labels, return_counts=True)
        length = len(labels)
        return sum([-(i / length) * np.log2(i / length) for i in counts])

    def _iterate(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 current_depth=0) -> DecisionNode:

        if len(labels) == 1:
            return DecisionNode(data, labels, is_leaf=True)

        impurity = self.criterion(labels)
        best_column, best_value = None, None
        for column, column_values in enumerate(data.T):
            for split_value in set(column_values):

                true_labels = labels[column_values >= split_value]
                false_labels = labels[column_values < split_value]

                if len(true_labels) == 0 or len(true_labels) == 0:
                    continue
                false_impurity = self.criterion(false_labels)
                true_impurity = self.criterion(true_labels)

                final_impurity = (len(false_labels) / len(labels)) * false_impurity + \
                                 (len(true_labels) / len(labels)) * true_impurity

                if final_impurity < impurity:
                    impurity = final_impurity
                    best_value = split_value
                    best_column = column

        if best_column is None or current_depth == self.max_depth:
            return DecisionNode(data, labels, is_leaf=True)
        else:
            false_data = data[(data[:, [best_column]]).flatten() < best_value]
            true_data = data[(data[:, [best_column]]).flatten() >= best_value]

            false_labels = labels[(data[:, [best_column]]).flatten() < best_value]
            true_labels = labels[(data[:, [best_column]]).flatten() >= best_value]

            return DecisionNode(data=data, labels=labels,
                                column=best_column, value=best_value,
                                false_branch=self._iterate(false_data, false_labels, current_depth + 1),
                                true_branch=self._iterate(true_data, true_labels, current_depth + 1))

    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, data: np.ndarray) -> list:
        y_pred = []
        for d in data:
            node = self.tree
            while True:
                if node.is_leaf:
                    y_pred.append(np.bincount(node.labels).argmax())
                    break
                if d[node.column] >= node.value:
                    node = node.true_branch
                else:
                    node = node.false_branch
        return y_pred


# def cross_val_splits(X: np.ndarray, Y: np.ndarray, *, folds: int = 5):
#     N = len(X)
#     for i in range(folds):
#         a, b = int(N * i / folds), int(N * (i + 1) / folds)
#         yield (np.concatenate((X[:a], X[b:])), np.concatenate((Y[:a], Y[b:]))), (X[a:b], Y[a:b])
