from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import librosa

# Load the model and other necessary tools
model = pickle.load(open('emotion_model.pkl', 'rb'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to save uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the folder exists

@app.route('/')
def index():
    return render_template('upload.html')  # Form for uploading audio files

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        emotion = predict_emotion(filepath)
        return render_template('result.html', prediction=emotion)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    features = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        features = np.hstack((features, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        features = np.hstack((features, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        features = np.hstack((features, mel))
    return features

def predict_emotion(file_path):
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
