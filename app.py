from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import librosa
import tempfile
import os

app = Flask(__name__)

# Carrega modelo TFLite
interpreter = tf.lite.Interpreter(model_path="C:\\sexta\\appe\\snore_application\\model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carrega scaler
scaler = joblib.load("C:\\sexta\\appe\\snore_application\\scaler.pkl")

def preprocess_audio(file_path, max_frames=128):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        stft = librosa.stft(audio, n_fft=512, hop_length=256)
        spectrogram = np.abs(stft)
        db_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)

        if db_spec.shape[1] < max_frames:
            pad_width = max_frames - db_spec.shape[1]
            db_spec = np.pad(db_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            db_spec = db_spec[:, :max_frames]

        flat = db_spec.flatten().reshape(1, -1)
        flat_scaled = scaler.transform(flat)
        return flat_scaled
    except Exception as e:
        print("Erro no preprocessamento:", e)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio"}), 400

    try:
        # Salvar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            features = preprocess_audio(tmp.name)
            os.unlink(tmp.name)  # apagar o arquivo temporário

        if features is None:
            return jsonify({"error": "Erro ao processar o áudio"}), 500

        # Rodar modelo TFLite
        interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = float(output[0][0])

        return jsonify({
            "roncando_probabilidade": prediction,
            "classe": "roncando" if prediction > 0.5 else "normal"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
