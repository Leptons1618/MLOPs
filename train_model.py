# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Data preprocessing and feature engineering

# Split data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Model evaluation and saving
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')