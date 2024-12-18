import argparse
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data():
    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_model(X_train, y_train):
    # Set up hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10]
    }
    # Initialize and train the model with GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    logging.info(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Random Forest model on the Iris dataset.')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model to disk.')
    args = parser.parse_args()

    X, y = load_data()
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logging.info(f'Model accuracy: {accuracy}')

    if args.save_model:
        joblib.dump(model, 'model.joblib')
        logging.info('Model saved to model.joblib')