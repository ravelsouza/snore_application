from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/teste', methods=['POST'])
def teste():
    return '<h1>Esse é o Teste de um modelo de Machine Leaning para detecção de ronco</h1>'



# Carrega modelo e scaler
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    print("Recebido áudio")
    file = request.files['audio']

    try:
        audio, sr = librosa.load(file, sr=16000)  # carrega áudio com 16kHz
        segment_duration = sr  # 1 segundo = 16000 amostras
        num_segments = len(audio) // segment_duration

        predictions = []
        prob = []

        for i in range(num_segments):
            start = i * segment_duration
            end = start + segment_duration
            segment = audio[start:end]

            # Gera espectrograma
            stft = librosa.stft(segment, n_fft=512, hop_length=256)
            spectrogram = np.abs(stft)
            db_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)

            # Ajusta para 128 frames
            if db_spec.shape[1] < 128:
                pad_width = 128 - db_spec.shape[1]
                db_spec = np.pad(db_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                db_spec = db_spec[:, :128]

            # Achata, normaliza e faz predição
            feat = db_spec.reshape(1, -1)
            feat_scaled = scaler.transform(feat)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], feat_scaled.astype(np.float32))
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            print("Mais uma predição")
            prob.append(float(output[0][0]))

            prediction = int(output[0][0] > 0.5)
            predictions.append(prediction)

        # Resumo da predição (opcional)
        percent_ronco = 100 * np.mean(predictions)
        resumo = "roncando" if percent_ronco > 50 else "normal"
        return jsonify({
            'prob': prob,
            'predictions': predictions,
            'percent_ronco': percent_ronco,
            'resumo': resumo
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
