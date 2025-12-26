import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def analyze_frame(frame):
    """
    Very simple logic:
    - Run eye detector on whole frame.
    - If we see 2+ eyes -> awake
    - If we see < 2 eyes -> drowsy
    Returns (is_drowsy: bool, ear_like: float)
    """
    try:
        if frame is None:
            print("FRAME IS NONE")
            return False, 0.0

        img = np.asarray(frame)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Resize for stability & speed
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔍 Detect eyes in whole image (no face ROI)
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(25, 25)
        )

        print("Eyes detected (global):", len(eyes))

        eyes_open = len(eyes) >= 2

        ear_like = 0.30 if eyes_open else 0.10
        is_drowsy = not eyes_open

        print("STATUS:", "AWAKE" if not is_drowsy else "DROWSY")

        return bool(is_drowsy), float(ear_like)

    except Exception as e:
        print("DETECTION ERROR:", e)
        return False, 0.0
