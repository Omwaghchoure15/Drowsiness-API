from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from detection import analyze_frame

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        image_base64 = data.get("image")

        if not image_base64:
            return jsonify({"error": "Empty image"}), 400

        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Decode failed"}), 400

        is_drowsy, ear = analyze_frame(frame)

        return jsonify({
            "drowsy": bool(is_drowsy),
            "ear": float(ear)
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "drowsy": False,
            "ear": 0.0
        }), 500
