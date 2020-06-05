import unittest

import pandas as pd
import numpy as np

from practical2.random_forest import RandomForestClassifier
from practical2.random_forest import f1_score, data_preprocess


class TestRandomForestClassifier(unittest.TestCase):

    def test_end_to_end(self):
        model = RandomForestClassifier()

        train_data = pd.read_csv("data/train.csv")
        train_data = train_data[:300]

        labels = train_data['label'].values
        x = np.array(train_data.drop('label', axis=1))
        y = labels

        model.fit(data_preprocess(x), y)
        y_predict = model.predict(data_preprocess(x))
        self.assertGreater(f1_score(y, y_predict), 0)

    def test_f1_Score(self):
        self.assertEqual(f1_score(np.array([1, 1, 1]), np.array([1, 1, 1])), 1)
        self.assertEqual(f1_score(np.array([1, 1, 1]), np.array([0, 0, 0])), 0)
        self.assertEqual(f1_score(np.array([1, 1, 1]), np.array([0, 1, 0])), 0.5)

    def test_data_preprocess(self):
        data = np.ones([5, 3])
        result = data_preprocess(data)
        self.assertEqual(data.shape[0], result.shape[0])
