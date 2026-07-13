import base64
import json
import pathlib
from typing import Any

import cv2
import numpy as np

_ROOT = pathlib.Path(__file__).parent
ENCODINGS_PATH = _ROOT / "reference_faces" / "encodings.json"
DETECTOR_MODEL = str(_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = str(_ROOT / "models" / "face_recognition_sface_2021dec.onnx")

# SFace cosine-similarity threshold. Above = same person.
# 0.363 is the recommended EER threshold from the paper; raise for stricter auth.
COSINE_THRESHOLD = 0.363


class FaceAuthenticator:
    def __init__(self) -> None:
        self._encodings: list[np.ndarray] = []
        self._names: list[str] = []
        ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL, "", (320, 320), score_threshold=0.6, nms_threshold=0.3
        )
        self._recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        self._load()

    # ── Enrollment ──────────────────────────────────────────────────────────

    def enroll_from_bytes(self, image_bytes: bytes, name: str = "person") -> dict[str, Any]:
        name = name.strip() or "person"
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Could not decode image."}

        encoding, error = self._extract_encoding(frame)
        if encoding is None:
            return {"success": False, "error": error}

        self._encodings.append(encoding)
        self._names.append(name)
        self._save()
        return {
            "success": True,
            "enrolledCount": len(self._encodings),
            "identities": self.enrolled_identities,
        }

    def clear(self) -> dict[str, Any]:
        self._encodings.clear()
        self._names.clear()
        self._save()
        return {"success": True, "enrolledCount": 0, "identities": []}

    # ── Verification ─────────────────────────────────────────────────────────

    def verify_from_base64(self, image_data: str) -> dict[str, Any]:
        if "," not in image_data:
            return {"authenticated": False, "error": "Invalid image data."}

        _, encoded = image_data.split(",", 1)
        try:
            binary = base64.b64decode(encoded)
        except Exception:
            return {"authenticated": False, "error": "Could not decode frame."}

        arr = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"authenticated": False, "error": "Could not parse frame."}

        return self._verify(frame)

    def _verify(self, frame: np.ndarray) -> dict[str, Any]:
        if not self._encodings:
            return {
                "authenticated": False,
                "faceDetected": False,
                "confidence": 0.0,
                "error": "No faces enrolled.",
            }

        live_enc, box, error = self._detect_and_encode(frame)
        if live_enc is None:
            return {"authenticated": False, "faceDetected": False, "confidence": 0.0}

        scores = [
            float(self._recognizer.match(live_enc, ref, cv2.FaceRecognizerSF_FR_COSINE))
            for ref in self._encodings
        ]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        confidence = round(max(0.0, min(1.0, best_score)), 3)
        matched_name = self._names[best_idx] if best_idx < len(self._names) else "unknown"

        h, w = frame.shape[:2]
        x, y, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        authenticated = best_score >= COSINE_THRESHOLD
        return {
            "authenticated": authenticated,
            "faceDetected": True,
            "confidence": confidence,
            "matchedName": matched_name if authenticated else None,
            "faceBox": {
                "top": round(y / h, 4),
                "right": round((x + bw) / w, 4),
                "bottom": round((y + bh) / h, 4),
                "left": round(x / w, 4),
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_encoding(self, frame: np.ndarray):
        """Returns (encoding, None) or (None, error_string)."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None, "No face detected in this image."
        aligned = self._recognizer.alignCrop(frame, faces[0])
        encoding = self._recognizer.feature(aligned).astype(np.float32)
        return encoding, None

    def _detect_and_encode(self, frame: np.ndarray):
        """Returns (encoding, box, None) or (None, None, error_string)."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None, None, "No face detected."
        box = faces[0][:4]
        aligned = self._recognizer.alignCrop(frame, faces[0])
        encoding = self._recognizer.feature(aligned).astype(np.float32)
        return encoding, box, None

    # ── Persistence ───────────────────────────────────────────────────────────

    @property
    def enrolled_count(self) -> int:
        return len(self._encodings)

    @property
    def enrolled_identities(self) -> list[str]:
        seen: dict[str, int] = {}
        for name in self._names:
            seen[name] = seen.get(name, 0) + 1
        return [f"{name} ({count} photo{'s' if count != 1 else ''})" for name, count in seen.items()]

    def _save(self) -> None:
        data = {
            "encodings": [enc.tolist() for enc in self._encodings],
            "names": self._names,
        }
        ENCODINGS_PATH.write_text(json.dumps(data))

    def _load(self) -> None:
        if not ENCODINGS_PATH.exists():
            return
        try:
            data = json.loads(ENCODINGS_PATH.read_text())
            self._encodings = [np.array(e, dtype=np.float32) for e in data.get("encodings", [])]
            self._names = data.get("names", ["person"] * len(self._encodings))
        except Exception:
            self._encodings = []
            self._names = []
