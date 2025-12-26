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
            print("No JSON received")
            return jsonify({"error": "No JSON received", "drowsy": False, "ear": 0.0}), 400

        image_base64 = data.get("imageBase64")

        if not image_base64:
            print("imageBase64 missing or empty:", image_base64)
            return jsonify({"error": "Empty imageBase64", "drowsy": False, "ear": 0.0}), 400

        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Could not decode image into frame")
            return jsonify({"error": "Decode failed", "drowsy": False, "ear": 0.0}), 400

        is_drowsy, ear = analyze_frame(frame)

        return jsonify({
            "drowsy": bool(is_drowsy),
            "ear": float(ear)
        }), 200

    except Exception as e:
        import traceback
        print("ERROR in /predict:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "drowsy": False, "ear": 0.0}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

