from flask import Flask, jsonify, render_template, request

from pose_detector import PoseDetector


app = Flask(__name__)
detector = PoseDetector()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/detect")
def detect_pose():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")

    if not image_data:
        return jsonify({"error": "Missing image payload."}), 400

    try:
        result = detector.detect_from_base64(image_data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
