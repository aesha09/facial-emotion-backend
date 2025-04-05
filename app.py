from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Analyze emotion using DeepFace
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        confidence = analysis[0]['emotion'][emotion]

        return jsonify({'emotion': emotion, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
