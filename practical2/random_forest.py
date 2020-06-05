import numpy as np
from practical2.decision_tree import DecisionTree


class RandomForestClassifier(object):
    def __init__(self,
                 n_trees=10,
                 criterion='gini',
                 max_depth=8,
                 data_size=0.8):
        """
        you can add as many parameters as you want to your classifier
        """
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.p = data_size
        self.forest = []

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        :param data: array of features for each point
        :param labels: array of labels for each point
        """
        for i in range(self.n_trees):
            row, column = data.shape
            tree = DecisionTree(self.max_depth, self.criterion)
            index_row = np.random.choice(row, round(row * self.p))
            tree.fit(data[index_row], labels[index_row])
            self.forest.append(tree)

    def predict(self, data: np.ndarray) -> np.ndarray:
        predicts = np.array(list(map(lambda tree: tree.predict(data), self.forest))).T
        predict = predicts.mean(axis=1)
        return np.rint(predict).astype(int)
        # raise NotImplementedError()


def f1_score(y_true: np.ndarray, y_predicted: np.ndarray):
    """
    only 0 and 1 should be accepted labels and 1 is the positive class
    """
    tp = sum((y_true == 1) & (y_predicted == 1))
    tn = sum((y_true == 0) & (y_predicted == 0))
    fn = sum((y_true == 1) & (y_predicted == 0))
    fp = sum((y_true == 0) & (y_predicted == 1))
    f1 = 0
    if tp or (fp and fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    return f1
    # assert set(y_true).union({1, 0}) == {1, 0}
    # raise NotImplementedError()


def data_preprocess(data: np.array) -> np.array:
    char_columns = []
    for i, value in enumerate(data[0]):
        if isinstance(value, str):
            char_columns.append(i)
    chars = set(data[:, char_columns].flatten())
    for i, char in enumerate(chars):
        data = np.where(data == char, i, data)
    return data.astype(int)
    # raise NotImplementedError()
