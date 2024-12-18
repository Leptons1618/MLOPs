from flask import Flask, request, jsonify
import joblib
import numpy as np
from train_model import preprocess_data

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    input_data_scaled = preprocess_data(input_data)
    prediction = model.predict(input_data_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)