const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const enrollLog = document.querySelector("#enroll-log");
const enrolledCount = document.querySelector("#enrolled-count");
const clearBtn = document.querySelector("#clear-btn");
const personNameInput = document.querySelector("#person-name");

async function fetchStatus() {
  try {
    const res = await fetch("/api/auth/status");
    const data = await res.json();
    enrolledCount.textContent = data.enrolledCount ?? "0";
  } catch (_) {}
}

async function uploadFile(file) {
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
    const res = await fetch("/api/auth/enroll", { method: "POST", body: formData });
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
  if (!confirm("Remove all enrolled face encodings?")) return;
  const res = await fetch("/api/auth/clear", { method: "POST" });
  const data = await res.json();
  enrolledCount.textContent = data.enrolledCount ?? 0;
  enrollLog.innerHTML = '<li class="enroll-log-empty">Cleared.</li>';
});

fetchStatus();
