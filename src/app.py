from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = np.array([[data['age'], data['sex'], data['cp'], data['trestbps'],
                          data['chol'], data['fbs'], data['restecg'], data['thalach'],
                          data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    output = int(prediction[0])
    return render_template('index.html', prediction_text=f'Prediction: {"Heart Disease" if output else "No Heart Disease"}')

if __name__ == '__main__':
    app.run(debug=True)
