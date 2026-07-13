const webcam = document.querySelector("#webcam");
const capture = document.querySelector("#capture");
const authOverlay = document.querySelector("#auth-overlay");
const videoShell = document.querySelector(".video-shell");
const cameraStatus = document.querySelector("#camera-status");
const authStatusBlock = document.querySelector("#auth-status-block");
const authRing = document.querySelector("#auth-ring");
const authRingIcon = document.querySelector("#auth-ring-icon");
const authStateLabel = document.querySelector("#auth-state-label");
const authNameLabel = document.querySelector("#auth-name-label");
const confidenceFill = document.querySelector("#confidence-fill");
const confidenceValue = document.querySelector("#confidence-value");
const enrolledBadge = document.querySelector("#enrolled-badge");

const overlayCtx = authOverlay.getContext("2d");
let isSending = false;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
    });
    webcam.srcObject = stream;
    await webcam.play();
    syncSizes();
    cameraStatus.textContent = "camera live";
    window.setInterval(sendFrame, 300);
  } catch (err) {
    cameraStatus.textContent = "camera blocked";
  }
}

async function fetchEnrolledCount() {
  try {
    const res = await fetch("/api/auth/status");
    const data = await res.json();
    if (enrolledBadge) enrolledBadge.textContent = data.enrolledCount ?? "—";
  } catch (_) {}
}

async function sendFrame() {
  if (isSending || webcam.readyState < 2) return;
  isSending = true;

  const ctx = capture.getContext("2d");
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(webcam, -capture.width, 0, capture.width, capture.height);
  ctx.restore();

  const image = capture.toDataURL("image/jpeg", 0.8);

  try {
    const res = await fetch("/api/auth/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const payload = await res.json();
    updateUI(payload);
  } catch (_) {
    setState("idle");
  } finally {
    isSending = false;
  }
}

let redirectTimer = null;

function updateUI(payload) {
  clearBox();

  if (payload.error && !payload.faceDetected) {
    setState("idle");
    setConfidence(0);
    cancelRedirect();
    return;
  }

  if (!payload.faceDetected) {
    setState("scanning");
    setConfidence(0);
    cancelRedirect();
    return;
  }

  setConfidence(payload.confidence ?? 0);
  drawFaceBox(payload.faceBox, payload.authenticated);
  setState(payload.authenticated ? "authenticated" : "denied", payload.matchedName ?? null);

  if (payload.authenticated) {
    scheduleRedirect();
  } else {
    cancelRedirect();
  }
}

function scheduleRedirect() {
  if (redirectTimer) return;
  redirectTimer = setTimeout(() => { window.location.href = "/"; }, 1500);
}

function cancelRedirect() {
  if (redirectTimer) {
    clearTimeout(redirectTimer);
    redirectTimer = null;
  }
}

function setState(state, name = null) {
  authStatusBlock.dataset.state = state;
  const labels = {
    idle: "waiting…",
    scanning: "scanning…",
    authenticated: "AUTHENTICATED",
    denied: "NOT RECOGNIZED",
  };
  const icons = { idle: "?", scanning: "◉", authenticated: "✓", denied: "✗" };
  authStateLabel.textContent = labels[state] ?? state;
  authRingIcon.textContent = icons[state] ?? "?";
  if (authNameLabel) {
    authNameLabel.textContent = state === "authenticated" && name ? name : "";
  }
}

function setConfidence(value) {
  const pct = Math.round(value * 100);
  confidenceFill.style.width = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
}

function drawFaceBox(box, authenticated) {
  if (!box) return;
  const frameW = capture.width || 640;
  const frameH = capture.height || 480;
  const shellW = videoShell.clientWidth || frameW;
  const shellH = videoShell.clientHeight || frameH;
  const scale = Math.max(shellW / frameW, shellH / frameH);
  const renderedW = frameW * scale;
  const renderedH = frameH * scale;
  const offsetX = (shellW - renderedW) / 2;
  const offsetY = (shellH - renderedH) / 2;

  const topPx = box.top * frameH * scale + offsetY;
  const rightPx = box.right * frameW * scale + offsetX;
  const bottomPx = box.bottom * frameH * scale + offsetY;
  const leftPx = box.left * frameW * scale + offsetX;

  const color = authenticated ? "#7dff85" : "#ff5e86";
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 3;
  overlayCtx.shadowColor = color;
  overlayCtx.shadowBlur = 10;
  overlayCtx.strokeRect(leftPx, topPx, rightPx - leftPx, bottomPx - topPx);
  overlayCtx.shadowBlur = 0;
}

function clearBox() {
  overlayCtx.clearRect(0, 0, authOverlay.width, authOverlay.height);
}

function syncSizes() {
  const w = webcam.videoWidth || 640;
  const h = webcam.videoHeight || 480;
  const shellW = videoShell.clientWidth || w;
  const shellH = videoShell.clientHeight || h;
  const pixelRatio = window.devicePixelRatio || 1;

  capture.width = w;
  capture.height = h;
  authOverlay.width = Math.round(shellW * pixelRatio);
  authOverlay.height = Math.round(shellH * pixelRatio);
  overlayCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

webcam.addEventListener("loadedmetadata", syncSizes);
window.addEventListener("resize", syncSizes);

fetchEnrolledCount();
startCamera();
