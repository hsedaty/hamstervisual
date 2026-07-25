const webcam = document.querySelector("#webcam");
const capture = document.querySelector("#capture");
const overlay = document.querySelector("#overlay");
const videoShell = document.querySelector(".video-shell");
const cameraStatus = document.querySelector("#camera-status");
const poseName = document.querySelector("#pose-name");
const poseConfidence = document.querySelector("#pose-confidence");
const stickerImage = document.querySelector("#sticker-image");
const debugPanel = document.querySelector("#debug-panel");
const devModeToggle = document.querySelector("#dev-mode-toggle");
let devMode = false;

const stickerMap = {
  "no-face": "/static/stickers/no-face.jpg",
  neutral: "/static/stickers/neutral.jpg",
  peace: "/static/stickers/peace.jpeg",
  "thumb up": "/static/stickers/thup.jpeg",
  "thumb down": "/static/stickers/thdown.jpeg",
  strong: "/static/stickers/strong.jpeg",
  shh: "/static/stickers/shh.jpeg",
  nerd: "/static/stickers/nerd.jpeg",
  love: "/static/stickers/love.jpeg",
};

let isSending = false;
const overlayContext = overlay.getContext("2d");

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 960 },
        height: { ideal: 720 },
        facingMode: "user",
      },
    });

    webcam.srcObject = stream;
    await webcam.play();
    syncCanvasSizes();
    cameraStatus.textContent = "camera live";
    window.setInterval(sendFrame, 350);
  } catch (error) {
    cameraStatus.textContent = "camera blocked";
    debugPanel.textContent = String(error);
  }
}

async function sendFrame() {
  if (isSending || webcam.readyState < 2) {
    return;
  }

  isSending = true;

  const context = capture.getContext("2d");
  context.save();
  context.scale(-1, 1);
  context.drawImage(webcam, -capture.width, 0, capture.width, capture.height);
  context.restore();

  const image = capture.toDataURL("image/jpeg", 0.72);

  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Detection failed.");
    }

    const pose = payload.pose in stickerMap ? payload.pose : "no-face";
    poseName.textContent = pose;
    poseConfidence.textContent = `${Math.round((payload.confidence || 0) * 100)}%`;
    stickerImage.src = stickerMap[pose];
    if (devMode) {
      drawOverlay(payload.overlay);
      debugPanel.textContent = JSON.stringify(payload.metrics, null, 2);
    } else {
      clearOverlay();
    }
  } catch (error) {
    clearOverlay();
    debugPanel.textContent = String(error);
  } finally {
    updateDevModeVisibility();
    isSending = false;
  }
}

function updateDevModeVisibility() {
  overlay.style.display = devMode ? "block" : "none";
  debugPanel.style.display = devMode ? "block" : "none";
}

function syncCanvasSizes() {
  const width = webcam.videoWidth || 640;
  const height = webcam.videoHeight || 480;
  const shellWidth = videoShell.clientWidth || width;
  const shellHeight = videoShell.clientHeight || height;
  const pixelRatio = window.devicePixelRatio || 1;

  capture.width = width;
  capture.height = height;
  overlay.width = Math.round(shellWidth * pixelRatio);
  overlay.height = Math.round(shellHeight * pixelRatio);
  overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function drawOverlay(overlayData) {
  clearOverlay();

  if (!overlayData) {
    return;
  }

  drawPoints(overlayData.face || [], "#7dff85", 5);
  for (const hand of overlayData.hands || []) {
    drawPoints(hand.points || [], "#7dff85", 5);
  }
}

function drawPoints(points, color, radius) {
  overlayContext.fillStyle = color;
  for (const point of points) {
    const mappedPoint = mapOverlayPoint(point);
    overlayContext.beginPath();
    overlayContext.arc(mappedPoint.x, mappedPoint.y, radius, 0, Math.PI * 2);
    overlayContext.fill();
  }
}

function mapOverlayPoint(point) {
  const frameWidth = capture.width || webcam.videoWidth || 640;
  const frameHeight = capture.height || webcam.videoHeight || 480;
  const shellWidth = videoShell.clientWidth || frameWidth;
  const shellHeight = videoShell.clientHeight || frameHeight;
  const scale = Math.max(shellWidth / frameWidth, shellHeight / frameHeight);
  const renderedWidth = frameWidth * scale;
  const renderedHeight = frameHeight * scale;
  const offsetX = (shellWidth - renderedWidth) / 2;
  const offsetY = (shellHeight - renderedHeight) / 2;

  return {
    x: point.x * scale + offsetX,
    y: point.y * scale + offsetY,
  };
}

function clearOverlay() {
  overlayContext.clearRect(0, 0, videoShell.clientWidth || overlay.width, videoShell.clientHeight || overlay.height);
}

webcam.addEventListener("loadedmetadata", syncCanvasSizes);
window.addEventListener("resize", syncCanvasSizes);

devModeToggle?.addEventListener("change", () => {
  devMode = devModeToggle.checked;
  updateDevModeVisibility();
});

updateDevModeVisibility();

document.querySelector("#sign-out-btn")?.addEventListener("click", async () => {
  await fetch("/api/auth/logout", { method: "POST" });
  window.location.href = "/auth";
});

startCamera();
