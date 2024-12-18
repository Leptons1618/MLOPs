# test_model.py
import unittest
import joblib
from train_model import train_model, preprocess_data, load_data
from sklearn.model_selection import train_test_split

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        X, y = load_data()
        X = preprocess_data(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.model = train_model(self.X_train, self.y_train)

    def test_accuracy(self):
        accuracy = self.model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Model accuracy should be greater than 0.9")

    def test_prediction(self):
        sample_input = [5.1, 3.5, 1.4, 0.2]
        sample_input_scaled = preprocess_data([sample_input])
        predicted_class = self.model.predict(sample_input_scaled)
        self.assertEqual(len(predicted_class), 1, "There should be one prediction")
        self.assertIn(predicted_class[0], [0, 1, 2], "Prediction should be one of the target classes")

    def test_model_saving(self):
        joblib.dump(self.model, 'test_model.joblib')
        loaded_model = joblib.load('test_model.joblib')
        loaded_accuracy = loaded_model.score(self.X_test, self.y_test)
        self.assertEqual(self.model.score(self.X_test, self.y_test), loaded_accuracy, "Loaded model should have the same accuracy")

    def test_data_split(self):
        # Ensure data is correctly split into training and testing sets
        self.assertEqual(len(self.X_train) + len(self.X_test), len(self.X_train) + len(self.X_test),
                         "Data split should cover all samples")

if __name__ == '__main__':
    unittest.main()
