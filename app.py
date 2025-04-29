from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
import joblib

scaler = joblib.load('scaler.pkl')
# Carregar scaler e modelo
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    X = np.array(data).reshape(1, -1)
    X_scaled = scaler.transform(X).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return jsonify({'prediction': output_data.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

