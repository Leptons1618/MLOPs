# test_model.py
import unittest
from train_model import train_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        self.data = load_iris()
        self.X = self.data.data
        self.y = self.data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = train_model(self.X_train, self.y_train)

    def test_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Model accuracy should be greater than 0.9")

    def test_prediction(self):
        sample_input = [5.1, 3.5, 1.4, 0.2]  # Modify this with a sample input matching your feature dimensions
        predicted_class = self.model.predict([sample_input])
        self.assertEqual(len(predicted_class), 1, "There should be one prediction")
        self.assertIn(predicted_class[0], [0, 1, 2], "Prediction should be one of the target classes")

    def test_data_split(self):
        # Ensure data is correctly split into training and testing sets
        self.assertEqual(len(self.X_train) + len(self.X_test), len(self.X), "Data split should cover all samples")

if __name__ == '__main__':
    unittest.main()
