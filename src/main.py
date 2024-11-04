'''

ai 서비스 백엔드 파이썬

'''
from flask import Flask, request, jsonify, render_template
from model.model import *
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 출처 허용

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# 파일 확장자 검사 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def test():
    return render_template('test.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['audioFile']
        if file and allowed_file(file.filename):
            # 파일을 메모리에서 바로 처리
            audio_data, sample_rate = librosa.load(file.stream, sr=None)  # 파일을 직접 읽어들임
            X_new = predict_data(audio_data, sample_rate)  # 필요한 전처리 수행
            
            # 모델 예측 수행
            prediction = model.predict(X_new)[0] 
            predicted_class = np.argmax(prediction)
            
            # 클래스별 확률 및 예측 결과
            probabilities = {classes[i]: float(prediction[i]) * 100 for i in range(len(classes))}
            result = {
                'predicted_class': classes[predicted_class],
                'probabilities': probabilities
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=True)
