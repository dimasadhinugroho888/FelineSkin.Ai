const MODEL_URL_CANDIDATES = ["./model/model.json", "/model/model.json"];
const META_URL = "./model/meta.json";
const EXPLAIN_API = "/api/explain";

const CAM_CONV_NODE = "PartitionedCall/Relu_16";
const LOGITS_NODE = "Identity";
const FC_WEIGHT_NODE = "unknown_46";

const diseaseMap = {
  Flea_Allergy: "Alergi kutu pada kucing",
  Health: "Kucing sehat",
  Ringworm: "Kurap pada kucing",
  Scabies: "Kudis pada kucing",
};

const state = {
  model: null,
  catDetector: null,
  classes: ["Flea_Allergy", "Health", "Ringworm", "Scabies"],
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  imageFile: null,
  lastPrediction: null,
  isValidImage: false,
};

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const probabilityList = document.getElementById("probabilityList");
const validationText = document.getElementById("validationText");
const camImage = document.getElementById("camImage");
const camStatus = document.getElementById("camStatus");
const aiExplanation = document.getElementById("aiExplanation");
const serviceType = document.getElementById("serviceType");
const locationInput = document.getElementById("locationInput");
const findMapBtn = document.getElementById("findMapBtn");
const mapFrame = document.getElementById("mapFrame");
const mapLink = document.getElementById("mapLink");

function setStatus(message, ready = false) {
  statusText.textContent = message;
  statusDot.classList.toggle("ready", ready);
}

function formatClassName(name) {
  return name.replaceAll("_", " ");
}

function updateImage(file) {
  state.imageFile = file;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  
  // Clear previous outputs
  predictionText.textContent = "-";
  confidenceText.textContent = "Confidence: -";
  probabilityList.innerHTML = "";
  camImage.classList.add("hidden");
  camStatus.textContent = "Heatmap belum tersedia.";
  aiExplanation.textContent = "Belum ada penjelasan.";
  
  validationText.style.color = "";
  validationText.textContent = "Memvalidasi gambar...";
  analyzeBtn.disabled = true;
  state.isValidImage = false;

  preview.onload = async () => {
    const result = await validateInputImage(preview);
    setValidationMessage(result);
    state.isValidImage = result.ok;
    
    if (state.model && state.isValidImage) {
      analyzeBtn.disabled = false;
    } else {
      analyzeBtn.disabled = true;
      if (!result.ok) {
        alert("Peringatan: Gambar tidak diizinkan! Harap upload foto kucing atau kulit kucing yang jelas.");
      }
    }
  };
}

function createProbabilityRows(probabilities) {
  const pairs = state.classes.map((name, idx) => ({
    name,
    prob: probabilities[idx] ?? 0,
  }));
  pairs.sort((a, b) => b.prob - a.prob);

  probabilityList.innerHTML = "";
  pairs.forEach((item) => {
    const row = document.createElement("div");
    row.className = "probability-item";
    row.innerHTML = `
      <div class="probability-head">
        <span>${formatClassName(item.name)}</span>
        <span>${(item.prob * 100).toFixed(2)}%</span>
      </div>
      <div class="bar"><span style="width:${Math.max(item.prob * 100, 1)}%"></span></div>
    `;
    probabilityList.appendChild(row);
  });
}

function preprocessImageToTensor(imgEl) {
  return tf.tidy(() => {
    const mean = tf.tensor1d(state.mean);
    const std = tf.tensor1d(state.std);

    let tensor = tf.browser.fromPixels(imgEl).toFloat().div(255.0);
    tensor = tf.image.resizeBilinear(tensor, [224, 224], true);
    tensor = tensor.sub(mean).div(std);
    return tensor.expandDims(0);
  });
}

function extractGrayPixels(imgEl, width = 224, height = 224) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgEl, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;

  const gray = new Float32Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  return { gray, width, height, canvas };
}

function closeupTextureCheck(imgEl) {
  const { gray, width, height } = extractGrayPixels(imgEl, 224, 224);

  let meanLap = 0;
  let sqLap = 0;
  let count = 0;
  let edgePixels = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const c = gray[idx];
      const up = gray[idx - width];
      const down = gray[idx + width];
      const left = gray[idx - 1];
      const right = gray[idx + 1];

      const lap = (up + down + left + right) - 4 * c;
      meanLap += lap;
      sqLap += lap * lap;
      count += 1;

      const gx = right - left;
      const gy = down - up;
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > 22) edgePixels += 1;
    }
  }

  meanLap /= count;
  const varLap = (sqLap / count) - (meanLap * meanLap);
  const edgeDensity = edgePixels / count;

  const textureLike = varLap > 180 && varLap < 2500 && edgeDensity > 0.06 && edgeDensity < 0.35;

  return { textureLike, varLap, edgeDensity };
}

async function detectCat(imgEl) {
  try {
    const preds = await state.catDetector.classify(imgEl, 5);

    const catLike = preds.some((p) => {
      const label = String(p.className || "").toLowerCase();
      return label.includes("cat") || label.includes("kitten") || label.includes("tabby") || label.includes("siamese");
    });

    return { isCat: catLike, preds };
  } catch (error) {
    return { isCat: true, preds: [], warning: error.message };
  }
}

async function validateInputImage(imgEl) {
  const catRes = await detectCat(imgEl);
  const textureRes = closeupTextureCheck(imgEl);

  // Check if it is explicitly detected as a dog or other non-cat animal
  const dogKeywords = [
    "dog", "puppy", "canine", "terrier", "retriever", "collie", "spaniel", 
    "hound", "mastiff", "poodle", "bulldog", "pug", "shepherd", "boxer", 
    "beagle", "corgi", "husky", "chihuahua", "pinscher", "pekingese", 
    "pomeranian", "shih", "toy dog", "working dog"
  ];

  const detectedDog = catRes.preds.some((p) => {
    const label = String(p.className || "").toLowerCase();
    return dogKeywords.some(keyword => label.includes(keyword)) && p.probability > 0.15;
  });

  if (detectedDog) {
    return {
      ok: false,
      severity: "bad",
      message: "Gambar terdeteksi sebagai anjing. Harap masukkan foto kucing atau kulit kucing saja.",
      details: { catRes, textureRes },
    };
  }

  // Check if it is explicitly detected as a human or human part
  const humanKeywords = [
    "human", "person", "man", "woman", "boy", "girl", "child", "baby", "face", 
    "groom", "gown", "t-shirt", "jersey", "suit", "coat", "jacket", "shirt", 
    "wig", "hair", "mask", "sunglass", "glasses", "spectacles", "bow tie", "necktie"
  ];

  const detectedHuman = catRes.preds.some((p) => {
    const label = String(p.className || "").toLowerCase();
    return humanKeywords.some(keyword => label.includes(keyword)) && p.probability > 0.15;
  });

  if (detectedHuman) {
    return {
      ok: false,
      severity: "bad",
      message: "Gambar terdeteksi sebagai manusia/bagian tubuh manusia. Harap masukkan foto kucing atau kulit kucing saja.",
      details: { catRes, textureRes },
    };
  }

  if (catRes.isCat) {
    return {
      ok: true,
      severity: "ok",
      message: "Kucing terdeteksi. Gambar valid untuk diproses.",
      details: { catRes, textureRes },
    };
  }

  if (textureRes.textureLike) {
    return {
      ok: true,
      severity: "warn",
      message: "Mode close-up/tekstur terdeteksi. Tetap diproses, tapi pastikan area kulit jelas.",
      details: { catRes, textureRes },
    };
  }

  return {
    ok: false,
    severity: "bad",
    message: "Gambar tidak dikenali sebagai kucing. Upload foto kulit kucing yang lebih jelas.",
    details: { catRes, textureRes },
  };
}

function setValidationMessage(result) {
  if (result.severity === "ok") {
    validationText.style.color = "#9cffbf";
  } else if (result.severity === "warn") {
    validationText.style.color = "#ffd27f";
  } else {
    validationText.style.color = "#ff9b9b";
  }
  validationText.textContent = result.message;
}

function dataURLFromTensorCam(baseImage, camMap2d) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(baseImage, 0, 0, 224, 224);

  const base = ctx.getImageData(0, 0, 224, 224);
  const out = base.data;

  for (let i = 0; i < camMap2d.length; i += 1) {
    const v = Math.min(1, Math.max(0, camMap2d[i]));
    const r = 255 * v;
    const b = 255 * (1 - v);

    const idx = i * 4;
    out[idx] = Math.round(out[idx] * 0.62 + r * 0.38);
    out[idx + 1] = Math.round(out[idx + 1] * 0.62 + 0 * 0.38);
    out[idx + 2] = Math.round(out[idx + 2] * 0.62 + b * 0.38);
  }

  ctx.putImageData(base, 0, 0);
  return canvas.toDataURL("image/png");
}

async function generateCamPreview(inputTensor, classIdx) {
  try {
    const outputs = await state.model.executeAsync(inputTensor, [CAM_CONV_NODE, FC_WEIGHT_NODE]);
    let convAct = null;
    let fcWeights = null;

    if (Array.isArray(outputs) && outputs.length >= 2) {
      [convAct, fcWeights] = outputs;
    } else {
      throw new Error("Output CAM tidak lengkap");
    }

    const selected = fcWeights.gather([classIdx]).reshape([1, 1, 1, -1]);
    let cam = convAct.mul(selected).sum(-1).relu();
    cam = cam.squeeze([0]);

    const min = cam.min();
    const max = cam.max();
    cam = cam.sub(min).div(max.sub(min).add(1e-7));
    cam = cam.expandDims(-1);
    cam = tf.image.resizeBilinear(cam, [224, 224], true).squeeze([-1]);

    const camData = await cam.data();

    tf.dispose([convAct, fcWeights, selected, cam, min, max]);

    const dataUrl = dataURLFromTensorCam(preview, camData);
    camImage.src = dataUrl;
    camImage.classList.remove("hidden");
    camStatus.textContent = "Heatmap berhasil dibuat (area merah paling berpengaruh).";
  } catch (error) {
    console.warn("CAM generation failed", error);
    camImage.classList.add("hidden");
    camStatus.textContent = `Heatmap belum bisa dibuat: ${error.message}`;
  }
}

function fallbackExplanation(label, confidence) {
  const simple = {
    Health: "Kucing tampak sehat. Tetap jaga kebersihan kulit, rutin grooming, dan kontrol berkala.",
    Flea_Allergy: "Kemungkinan alergi kutu. Periksa adanya kutu, bersihkan lingkungan, dan konsultasi dokter jika gatal berat.",
    Ringworm: "Kemungkinan kurap. Isolasi sementara, jaga area tetap bersih dan kering, lalu konsultasikan terapi antijamur.",
    Scabies: "Kemungkinan kudis. Hindari kontak dekat dengan hewan lain dan segera periksa ke dokter hewan untuk terapi tepat.",
  };

  const msg = simple[label] ?? "Belum ada penjelasan khusus untuk hasil ini.";
  return `${msg}\n\nConfidence model: ${confidence.toFixed(2)}%\nCatatan: Ini bukan diagnosis medis definitif.`;
}

async function fetchAIExplanation(label, confidence) {
  const diseaseName = diseaseMap[label] || formatClassName(label);

  try {
    const res = await fetch(EXPLAIN_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        diseaseKey: label,
        diseaseName,
        confidence,
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data?.explanation) {
      return data.explanation;
    }

    throw new Error("Response explanation kosong");
  } catch (error) {
    return `${fallbackExplanation(label, confidence)}\n\n(Mode fallback lokal: ${error.message})`;
  }
}

function updateMap() {
  const service = serviceType.value || "dokter hewan";
  const loc = (locationInput.value || "").trim();
  const query = loc ? `${service} terdekat di ${loc}` : `${service} terdekat`;

  const embedUrl = `https://www.google.com/maps?q=${encodeURIComponent(query)}&output=embed`;
  const searchUrl = `https://www.google.com/maps/search/${encodeURIComponent(query)}`;

  mapFrame.src = embedUrl;
  mapLink.href = searchUrl;
}

async function predict() {
  if (!state.model || !state.imageFile || !state.isValidImage) {
    alert("Peringatan: Gambar tidak diizinkan! Harap upload foto kucing atau kulit kucing yang jelas sebelum memulai analisis.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Menganalisis...";

  try {
    const inputTensor = preprocessImageToTensor(preview);
    let logits = await state.model.executeAsync(inputTensor, LOGITS_NODE);
    if (Array.isArray(logits)) {
      logits = logits[0];
    }

    const probs = tf.softmax(logits);
    const probData = Array.from(await probs.data());

    const bestIdx = probData.indexOf(Math.max(...probData));
    const bestName = state.classes[bestIdx] || `Class ${bestIdx}`;
    const confidence = probData[bestIdx] * 100;

    state.lastPrediction = { bestIdx, bestName, confidence, probs: probData };

    predictionText.textContent = diseaseMap[bestName] || formatClassName(bestName);
    confidenceText.textContent = `Confidence: ${confidence.toFixed(2)}%`;

    if (confidence < 50) {
      confidenceText.textContent += " (rendah, cek ulang gambar)";
    }

    createProbabilityRows(probData);
    await generateCamPreview(inputTensor, bestIdx);

    aiExplanation.textContent = "Mengambil penjelasan AI...";
    aiExplanation.textContent = await fetchAIExplanation(bestName, confidence);

    tf.dispose([inputTensor, logits, probs]);
  } catch (error) {
    console.error(error);
    setStatus(`Gagal inferensi: ${error.message}`, false);
  } finally {
    analyzeBtn.textContent = "Analisis Sekarang";
    analyzeBtn.disabled = !(state.imageFile && state.isValidImage);
  }
}

async function loadMeta() {
  try {
    const res = await fetch(META_URL);
    if (!res.ok) return;

    const meta = await res.json();
    if (Array.isArray(meta.classes) && meta.classes.length > 0) state.classes = meta.classes;
    if (Array.isArray(meta.mean) && meta.mean.length === 3) state.mean = meta.mean;
    if (Array.isArray(meta.std) && meta.std.length === 3) state.std = meta.std;
  } catch (error) {
    console.warn("Meta not loaded, fallback defaults", error);
  }
}

async function loadModel() {
  if (window.location.protocol === "file:") {
    throw new Error("Aplikasi dibuka via file://. Jalankan lewat server lokal (http://127.0.0.1:8080). ");
  }

  setStatus("Menyiapkan engine AI...");
  await tf.ready();

  setStatus("Mengunduh model AI...");
  await loadMeta();

  let loaded = false;
  let lastError = null;

  for (const candidate of MODEL_URL_CANDIDATES) {
    try {
      const modelUrl = new URL(candidate, window.location.href).toString();
      const res = await fetch(modelUrl, { method: "GET" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} untuk ${candidate}`);
      }
      state.model = await tf.loadGraphModel(modelUrl);
      loaded = true;
      break;
    } catch (error) {
      lastError = error;
    }
  }

  if (!loaded) {
    throw new Error(`Model tidak bisa diunduh. Cek server/path model. Detail: ${lastError?.message ?? "unknown"}`);
  }

  setStatus("Memuat model validasi kucing...");
  state.catDetector = await mobilenet.load({ version: 2, alpha: 1.0 });

  setStatus("Model siap digunakan", true);
  analyzeBtn.disabled = !(state.imageFile && state.isValidImage);
}

function setupUploadHandlers() {
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) updateImage(file);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      updateImage(file);
    }
  });
}

function setupSidebar() {
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebarClose = document.getElementById("sidebarClose");
  const sidebarOverlay = document.getElementById("sidebarOverlay");

  if (!sidebar || !sidebarToggle || !sidebarClose || !sidebarOverlay) return;

  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.add("open");
    sidebarOverlay.classList.add("active");
  });

  const closeSidebar = () => {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("active");
  };

  sidebarClose.addEventListener("click", closeSidebar);
  sidebarOverlay.addEventListener("click", closeSidebar);
}

function setupFelinaChatbot() {
  const widget = document.getElementById("felinaWidget");
  const toggleBtn = document.getElementById("felinaToggle");
  const chatWindow = document.getElementById("felinaChatWindow");
  const closeBtn = document.getElementById("felinaClose");
  const chatBody = document.getElementById("felinaChatBody");
  const inputEl = document.getElementById("felinaInput");
  const sendBtn = document.getElementById("felinaSend");
  const badge = toggleBtn?.querySelector(".felina-toggle-badge");

  if (!widget || !toggleBtn || !chatWindow || !closeBtn || !chatBody || !inputEl || !sendBtn) return;

  let messagesContext = [];
  let isSending = false;

  // Toggle chat window
  toggleBtn.addEventListener("click", () => {
    chatWindow.classList.toggle("hidden");
    if (!chatWindow.classList.contains("hidden")) {
      if (badge) badge.classList.add("hidden-badge");
      inputEl.focus();
      scrollToBottom();
    }
  });

  // Close chat window
  closeBtn.addEventListener("click", () => {
    chatWindow.classList.add("hidden");
  });

  // Scroll to bottom helper
  function scrollToBottom() {
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  // Create message bubble
  function appendMessage(role, text) {
    const isBot = role === "assistant";
    const msgDiv = document.createElement("div");
    msgDiv.className = `felina-msg ${isBot ? "felina-msg-bot" : "felina-msg-user"}`;

    const avatarHtml = isBot ? `<img src="logo felina.png" alt="Felina" class="felina-bubble-avatar" />` : "";
    
    // Convert newlines to breaks safely, and bold tags
    let formattedText = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br>")
      .replace(/&lt;strong&gt;/g, "<strong>")
      .replace(/&lt;\/strong&gt;/g, "</strong>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    msgDiv.innerHTML = `
      ${avatarHtml}
      <div class="felina-bubble">${formattedText}</div>
    `;

    chatBody.appendChild(msgDiv);
    scrollToBottom();
  }

  // Show/Hide typing indicator
  let typingIndicatorDiv = null;
  function showTypingIndicator() {
    if (typingIndicatorDiv) return;
    typingIndicatorDiv = document.createElement("div");
    typingIndicatorDiv.className = "felina-msg felina-msg-bot felina-typing";
    typingIndicatorDiv.innerHTML = `
      <img src="logo felina.png" alt="Felina" class="felina-bubble-avatar" />
      <div class="felina-bubble">
        <span></span>
        <span></span>
        <span></span>
      </div>
    `;
    chatBody.appendChild(typingIndicatorDiv);
    scrollToBottom();
  }

  function hideTypingIndicator() {
    if (typingIndicatorDiv) {
      typingIndicatorDiv.remove();
      typingIndicatorDiv = null;
    }
  }

  // Send message function
  async function sendMessage(text) {
    if (!text.trim() || isSending) return;

    isSending = true;
    inputEl.value = "";
    inputEl.disabled = true;
    sendBtn.disabled = true;

    // Append user message
    appendMessage("user", text);
    messagesContext.push({ role: "user", content: text });

    showTypingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: messagesContext,
        }),
      });

      hideTypingIndicator();

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const reply = data.reply || "Maaf, Felina sedang bingung. Bisa ulangi pertanyaannya? 🐱";
      
      appendMessage("assistant", reply);
      messagesContext.push({ role: "assistant", content: reply });

    } catch (err) {
      console.error("Chat error:", err);
      hideTypingIndicator();
      appendMessage("assistant", "Koneksi terganggu. Silakan pastikan server aktif dan coba lagi beberapa saat ya! 🐾");
    } finally {
      isSending = false;
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.focus();
      scrollToBottom();
    }
  }

  // Form submits
  sendBtn.addEventListener("click", () => {
    sendMessage(inputEl.value);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputEl.value);
    }
  });

  // Template Quick Questions
  chatBody.addEventListener("click", (e) => {
    const btn = e.target.closest(".felina-template-btn");
    if (btn) {
      const query = btn.getAttribute("data-q");
      if (query) {
        // Remove templates container after first interaction to save space
        const templates = chatBody.querySelector(".felina-templates");
        if (templates) templates.remove();
        
        sendMessage(query);
      }
    }
  });
}

async function init() {
  setupSidebar();
  setupFelinaChatbot();
  setupUploadHandlers();
  analyzeBtn.addEventListener("click", predict);
  findMapBtn.addEventListener("click", updateMap);
  updateMap();

  try {
    await loadModel();
  } catch (error) {
    console.error(error);
    setStatus(`Gagal load model: ${error.message}`, false);
  }
}

init();
