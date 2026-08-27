const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const enrollLog = document.querySelector("#enroll-log");
const enrolledCount = document.querySelector("#enrolled-count");
const clearBtn = document.querySelector("#clear-btn");
const personNameInput = document.querySelector("#person-name");
const pageConfig = {
  staticPreview: document.body?.dataset.staticPreview === "true",
  apiBase: document.body?.dataset.apiBase || "",
};

function getApiUrl(path) {
  return pageConfig.apiBase ? `${pageConfig.apiBase}${path}` : path;
}

function appendPreviewMessage(message) {
  if (enrollLog.querySelector(".enroll-log-empty")) enrollLog.innerHTML = "";
  const li = document.createElement("li");
  li.className = "enroll-log-item enroll-log-item--err";
  li.textContent = message;
  enrollLog.prepend(li);
}

async function fetchStatus() {
  if (pageConfig.staticPreview) {
    enrolledCount.textContent = "local only";
    return;
  }

  try {
    const res = await fetch(getApiUrl("/api/auth/status"));
    const data = await res.json();
    enrolledCount.textContent = data.enrolledCount ?? "0";
  } catch (_) {}
}

async function uploadFile(file) {
  if (pageConfig.staticPreview) {
    appendPreviewMessage(`${file.name} — GitHub Pages cannot call the Python enroll API. Run the Flask app locally to add faces.`);
    return;
  }

  const name = personNameInput ? personNameInput.value.trim() || "person" : "person";

  const li = document.createElement("li");
  li.className = "enroll-log-item enroll-log-item--pending";
  li.textContent = `${file.name} (${name}) — uploading…`;
  if (enrollLog.querySelector(".enroll-log-empty")) enrollLog.innerHTML = "";
  enrollLog.prepend(li);

  const formData = new FormData();
  formData.append("photo", file);
  formData.append("name", name);

  try {
    const res = await fetch(getApiUrl("/api/auth/enroll"), { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
      li.className = "enroll-log-item enroll-log-item--ok";
      const summary = (data.identities || []).join(" · ");
      li.textContent = `${file.name} — enrolled as "${name}" · ${summary}`;
      enrolledCount.textContent = data.enrolledCount;
    } else {
      li.className = "enroll-log-item enroll-log-item--err";
      li.textContent = `${file.name} — ${data.error}`;
    }
  } catch (err) {
    li.className = "enroll-log-item enroll-log-item--err";
    li.textContent = `${file.name} — upload failed`;
  }
}

async function uploadFiles(files) {
  for (const file of files) {
    await uploadFile(file);
  }
}

fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drop-zone--over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drop-zone--over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drop-zone--over");
  const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
  uploadFiles(files);
});

clearBtn.addEventListener("click", async () => {
  if (pageConfig.staticPreview) {
    appendPreviewMessage("GitHub Pages preview mode cannot clear or upload enrolled faces.");
    return;
  }

  if (!confirm("Remove all enrolled face encodings?")) return;
  const res = await fetch(getApiUrl("/api/auth/clear"), { method: "POST" });
  const data = await res.json();
  enrolledCount.textContent = data.enrolledCount ?? 0;
  enrollLog.innerHTML = '<li class="enroll-log-empty">Cleared.</li>';
});

fetchStatus();
