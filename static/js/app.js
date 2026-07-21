/* Inkstead frontend — drives the 4-step wizard against the Flask API. */

const state = {
  step: 1,
  samples: [],
  styleProfile: null,
  lastGeneration: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 3200);
}

function goToStep(n) {
  state.step = n;
  $$(".panel").forEach((p) => p.classList.remove("active"));
  $(`#panel-${n}`).classList.add("active");
  $$(".step-tab").forEach((t) => t.classList.toggle("active", Number(t.dataset.step) === n));
}

$$("[data-goto]").forEach((btn) =>
  btn.addEventListener("click", () => goToStep(Number(btn.dataset.goto)))
);
$$(".step-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    const n = Number(tab.dataset.step);
    // Only allow jumping to steps that are already unlocked.
    if (n === 1) return goToStep(1);
    if (n === 2) return; // skip step 2 for template mode
    if (n === 3 && state.styleProfile) return goToStep(3);
    if (n === 4 && state.lastGeneration) return goToStep(4);
  })
);

/* ---------------- Step 1: upload ---------------- */

const dropzone = $("#dropzone");
const fileInput = $("#fileInput");

dropzone.addEventListener("click", () => fileInput.click());
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); })
);
dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length) uploadSamples(files);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadSamples(fileInput.files[0]);
});

async function uploadSamples(file) {
  const form = new FormData();
  form.append("template", file);

  $("#uploadStatus").innerHTML = `<span class="spinner"></span> Processing template…`;
  try {
    const res = await fetch("/api/template/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    // Mock style profile so step 3 allows generation
    state.styleProfile = { recommended_font: "neat", slant_deg: 0 };
    $("#uploadStatus").textContent = `Template processed! ${data.chars_extracted} characters extracted.`;
    $("#toStep3Direct").disabled = false;
    toast(`Template loaded successfully.`);
  } catch (err) {
    $("#uploadStatus").textContent = "";
    toast(err.message, true);
  }
  fileInput.value = "";
}

function renderSampleGrid() {
  const grid = $("#sampleGrid");
  grid.innerHTML = state.samples
    .map(
      (s) => `
      <div class="sample-card">
        <img src="${s.preview_url}" alt="Preprocessed sample">
        <div class="tag">deskewed ${s.skew_angle_deg}°</div>
      </div>`
    )
    .join("");
}

$("#toStep3Direct")?.addEventListener("click", () => goToStep(3));

/* ---------------- Step 2: style extraction ---------------- */

$("#runExtract").addEventListener("click", async () => {
  const body = $("#styleBody");
  body.innerHTML = `<p class="hint"><span class="spinner"></span> Measuring slant, stroke weight, spacing, and training the style encoder…</p>`;
  try {
    const res = await fetch("/api/extract-style", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Extraction failed");

    state.styleProfile = data.style_profile;
    state.latentModifiers = data.latent_modifiers;
    if (data.style_profile.recommended_font) {
      $("#fontSelect").value = data.style_profile.recommended_font;
    }
    renderStyleMetrics(data.style_profile, data.latent_modifiers);
    $("#toStep3").disabled = false;
    toast("Style profile extracted.");

  } catch (err) {
    body.innerHTML = `<button class="btn primary" id="runExtractRetry">Try again</button>`;
    $("#runExtractRetry")?.addEventListener("click", () => $("#runExtract").click());
    toast(err.message, true);
  }
});

$("#toStep3").addEventListener("click", () => goToStep(3));

function renderStyleMetrics(profile, latent) {
  const cards = [
    ["Slant", `${profile.slant_deg.toFixed(1)}°`],
    ["Stroke width", `${profile.stroke_width_px.toFixed(1)}px`],
    ["Letter spacing", `${profile.letter_spacing.toFixed(1)}px`],
    ["Baseline jitter", `${profile.baseline_jitter.toFixed(1)}px`],
    ["Ink density", `${(profile.ink_density * 100).toFixed(1)}%`],
    ["Roughness", profile.roughness.toFixed(2)],
    ["Samples used", profile.n_samples],
    ["Learned flow", latent.flow_mod.toFixed(2)],
  ];
  $("#styleBody").innerHTML = `
    <div class="metric-grid">
      ${cards
        .map(
          ([label, value]) => `
        <div class="metric-card">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>`
        )
        .join("")}
    </div>
    <p class="hint" style="margin-top:1rem">These numbers, plus a 16-dimensional latent code from a small PyTorch autoencoder trained on your samples' glyph shapes, drive every character placed on the generated page.</p>
  `;
}

/* ---------------- Step 3: compose ---------------- */

const fontSizeRange = $("#fontSizeRange");
fontSizeRange.addEventListener("input", () => {
  $("#fontSizeLabel").textContent = `${fontSizeRange.value}px`;
});

$("#runGenerate").addEventListener("click", async () => {
  const text = $("#textInput").value.trim();
  if (!text) return toast("Type something first.", true);

  const hexToRgb = (hex) => {
    const m = hex.replace("#", "");
    return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
  };

  const payload = {
    text,
    font: $("#fontSelect").value,
    page_size: $("#pageSizeSelect").value,
    font_size: Number(fontSizeRange.value),
    ink_color: hexToRgb($("#inkColor").value),
    ruled_lines: $("#ruledLines").checked,
  };

  const btn = $("#runGenerate");
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Writing…`;

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Generation failed");

    state.lastGeneration = data;
    renderPreview(data);
    goToStep(4);
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
});

/* ---------------- Step 4: preview & export ---------------- */

function renderPreview(data) {
  $("#pagePreview").innerHTML = data.page_previews
    .map((url) => `<img src="${url}" alt="Generated handwriting page">`)
    .join("");
  $("#downloadPng").href = data.png_urls[0];
  $("#downloadPdf").href = data.pdf_url;
}
