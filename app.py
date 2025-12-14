import os
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2

app = Flask(__name__)
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

def encode_face(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if len(encodings) == 0:
        return None
    return encodings[0]

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "No frame uploaded"}), 400
    frame = request.files["frame"].read()
    encoding = encode_face(frame)
    if encoding is None:
        return jsonify({"status": "error", "message": "No face detected"}), 400
    np.save(os.path.join(MODEL_FOLDER, f"{username}.npy"), encoding)
    return jsonify({"status": "success", "message": f"User {username} registered"})

@app.route("/verify", methods=["POST"])
def verify():
    username = request.form.get("username")
    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "No frame uploaded"}), 400
    frame = request.files["frame"].read()
    encoding = encode_face(frame)
    if encoding is None:
        return jsonify({"status": "error", "message": "No face detected"}), 400
    filepath = os.path.join(MODEL_FOLDER, f"{username}.npy")
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "User not registered"}), 400
    saved_encoding = np.load(filepath)
    matches = face_recognition.compare_faces([saved_encoding], encoding)
    if matches[0]:
        return jsonify({"status": "success", "message": "User verified"})
    else:
        return jsonify({"status": "error", "message": "Face does not match"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
