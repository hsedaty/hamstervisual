# Hamster Visual

Basic Flask web app that reads webcam frames in the browser, sends them to a Python backend, and swaps doodle hamster stickers based on simple face and hand pose heuristics.

## Stack

- Flask for the web server
- OpenCV for frame decoding and image handling
- MediaPipe for face and hand landmarks
- Plain HTML, CSS, and JavaScript for the UI

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open http://127.0.0.1:5000 in the browser and allow webcam access.

## Publish on GitHub Pages

GitHub Pages can only host the static frontend from this repository. The Python Flask routes and OpenCV or MediaPipe APIs in `app.py`, `face_auth.py`, and `pose_detector.py` do not run on Pages.

This repo now includes Pages-ready entry points in the repository root:

- `index.html`
- `auth/index.html`
- `auth/enroll/index.html`

To publish them:

1. Push the repository to GitHub.
2. Open Settings > Pages.
3. Set Source to `Deploy from a branch`.
4. Choose branch `main`.
5. Choose folder `/ (root)`.
6. Save.

The published site will show the UI and webcam preview, but live pose detection and face enrollment or verification still require running the Flask app locally.

## Current pose labels

- `strong`: one hand high, one low, wide spread
- `shh`: index finger close to the mouth
- `wink`: one eye more closed than the other and mouth open
- `nerd`: index finger close to an eye area
- `love`: two hands close together near the face
- `neutral`: face detected but no pose matched
- `no-face`: no face detected

## Replace the sticker art

The app ships with simple SVG placeholders in `static/stickers/`. Replace those files with your own hamster doodles and keep the same filenames:

- `no-face.svg`
- `neutral.svg`
- `strong.svg`
- `shh.svg`
- `wink.svg`
- `nerd.svg`
- `love.svg`

## Tuning detection

The pose rules live in `pose_detector.py`. The current thresholds are intentionally simple so you can iterate quickly against your own camera framing and sticker poses.
