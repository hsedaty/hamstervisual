import os

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from face_auth import FaceAuthenticator
from pose_detector import PoseDetector


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hamster-dev-secret-change-me")
detector = PoseDetector()
authenticator = FaceAuthenticator()


@app.get("/")
def index() -> str:
    if not session.get("authenticated"):
        return redirect(url_for("auth_page"))
    return render_template("index.html", matched_name=session.get("matched_name", ""))


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


# ── Auth pages ───────────────────────────────────────────────────────────────

@app.get("/auth")
def auth_page() -> str:
    return render_template("auth.html")


@app.get("/auth/enroll")
def enroll_page() -> str:
    return render_template("enroll.html")


# ── Auth API ─────────────────────────────────────────────────────────────────

@app.get("/api/auth/status")
def auth_status():
    return jsonify({
        "enrolledCount": authenticator.enrolled_count,
        "identities": authenticator.enrolled_identities,
    })


@app.post("/api/auth/enroll")
def auth_enroll():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"success": False, "error": "No photo uploaded."}), 400

    name = request.form.get("name", "person")
    result = authenticator.enroll_from_bytes(photo.read(), name=name)
    status = 200 if result["success"] else 422
    return jsonify(result), status


@app.post("/api/auth/verify")
def auth_verify():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")

    if not image_data:
        return jsonify({"error": "Missing image payload."}), 400

    result = authenticator.verify_from_base64(image_data)

    if result.get("authenticated"):
        session["authenticated"] = True
        session["matched_name"] = result.get("matchedName", "")

    return jsonify(result)


@app.post("/api/auth/logout")
def auth_logout():
    session.clear()
    return jsonify({"success": True})


@app.post("/api/auth/clear")
def auth_clear():
    return jsonify(authenticator.clear())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
