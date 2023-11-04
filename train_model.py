# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data  # Features (sepal length, sepal width, petal length, and petal width)
y = data.target  # Target labels (0 for setosa, 1 for versicolor, 2 for virginica)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Model evaluation and saving
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')