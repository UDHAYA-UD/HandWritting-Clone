# Inkstead — AI Handwriting Style Generator

Upload a few photos of someone's handwriting, learn its style, type any text,
and get back a page (PNG/PDF) written to look like that hand.

```
Upload samples → Preprocess (OpenCV) → Extract style (CV + PyTorch) →
Type text → Generate page → Preview → Download PNG/PDF
```

---

## 1. Quick start (5–10 minutes)

**Requirements:** Python 3.10+, ~500MB free disk (mostly for PyTorch), internet
access for the one-time `pip install`.

```bash
# 1. Go into the project folder
cd Handwriting-Generator

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# ^ If this pulls a large CUDA build of torch and you don't have a GPU / want
#   a faster, smaller install, use the CPU-only wheel instead:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser. That's it — no database, no
model checkpoints to download, no external API keys.

### Using it
1. **Samples** — drag in 2–5 images of handwriting (photos or scans are fine).
2. **Style** — click "Analyze samples"; you'll see the measured slant, stroke
   width, spacing, jitter, and a learned "flow" value.
3. **Compose** — type the text you want written, pick a handwriting feel,
   page size, letter size, ink color, and whether to show ruled lines.
4. **Export** — preview the generated page(s) and download as PNG or PDF.

Everything runs locally; uploaded samples and generated pages never leave
your machine.

---

## 2. Project structure

```
Handwriting-Generator/
├── app.py                       # Flask routes/API
├── config.py                    # backend switch, upload limits
├── requirements.txt
├── models/
│   ├── style_autoencoder.py     # PyTorch conv autoencoder (learned style latent)
│   ├── handwriting_renderer.py  # Pillow-based style-conditioned page renderer
│   └── neural_hooks.py          # integration seam for real DeepWriting/DiffusionPen
├── utils/
│   ├── preprocessing.py         # OpenCV: grayscale, denoise, binarize, deskew, crop
│   ├── style_extraction.py      # slant/stroke/spacing/jitter/roughness metrics
│   └── export.py                # PNG + multi-page PDF export
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── fonts/                   # base handwriting fonts (OFL-licensed) + UI font
├── templates/
│   └── index.html
├── uploads/                     # per-session uploaded samples (gitignored)
└── output/                      # per-session generated pages (gitignored)
```

---

## 3. How generation actually works (read this before assuming "DeepWriting" = a downloaded checkpoint)

The spec names **DeepWriting** as the primary pretrained model and
**DiffusionPen** as a research alternative. Both are real published models,
but both ship as standalone research repos with their own multi-hundred-MB
to multi-GB pretrained checkpoints, bespoke training code, and (for
DiffusionPen) diffusion sampling that takes real GPU time per page. Vendoring
either of those directly would break the "runnable on localhost in ~15
minutes, no GPU required" goal of this build, and redistributing someone
else's checkpoint isn't something to fake.

So this project ships a **fully working, from-scratch pipeline** that mirrors
the same conceptual stages DeepWriting uses (analyze real handwriting →
encode style → condition generation on it) using components that are honest
about what they are:

1. **`utils/preprocessing.py`** — real OpenCV/NumPy pipeline: grayscale,
   bilateral denoise, Otsu binarization, skew estimation via `minAreaRect`,
   deskewing, and cropping to ink content.
2. **`utils/style_extraction.py`** — classical, explainable metrics computed
   straight from the binarized ink: slant angle, stroke width & variance
   (via distance transform), letter spacing (via connected components),
   baseline jitter, ink density, and stroke roughness.
3. **`models/style_autoencoder.py`** — a small **PyTorch** convolutional
   autoencoder that is genuinely trained (forward + backward passes,
   reconstruction loss decreasing) for a couple of seconds on the glyph
   patches cropped from your own uploads, right at extraction time. Its
   16-D latent code captures texture the hand-coded metrics can't, and
   modulates jitter/slant-variance/stroke-variance in the renderer — so two
   people's samples measurably produce different output beyond the
   classical stats alone.
4. **`models/handwriting_renderer.py`** — a Pillow-based renderer that places
   every character individually, each with its own slant, baseline offset,
   size jitter, and spacing, all derived from steps 2–3, on top of a base
   handwriting-style font (glyph shapes only — the *arrangement* is what
   simulates the learned style).
5. **`utils/export.py`** — PNG + multi-page PDF export.

This is a legitimate, if intentionally lightweight, style-conditioned
handwriting synthesis pipeline — not a stub that just echoes the input text
in one fixed font.

### Using the real DeepWriting / DiffusionPen models

If you want actual neural stroke-sequence generation instead of the
procedural renderer above:

1. Clone the model's research repo (search GitHub for the paper's official
   implementation — e.g. "DeepWriting: Making Digital Ink Editable via
   Deep Generative Modeling" or "DiffusionPen") and follow its own
   `requirements.txt`/environment setup. These are separate projects with
   their own licenses; review those before use.
2. Download the pretrained checkpoint the repo's README points to (these
   are typically hosted on Google Drive or the authors' own servers, not
   pip-installable).
3. Implement `ExternalHandwritingModel.load()` and `.generate_strokes()` in
   `models/neural_hooks.py` to load that checkpoint and call the model's
   sampling function, returning either a stroke-point sequence
   (DeepWriting) or a rendered image tensor (DiffusionPen).
4. Set `HANDWRITING_BACKEND=deepwriting` (or `diffusionpen`) as an
   environment variable, and wire `app.py`'s `/api/generate` route to call
   your hook instead of `render_multi_page()` when that env var is set.
5. If the model outputs stroke points rather than pixels, render them with
   Pillow's `ImageDraw.line()` along consecutive points instead of drawing
   font glyphs — the rest of the pipeline (style extraction, export) is
   reusable as-is.

This will need a GPU for reasonable DiffusionPen inference times; DeepWriting
runs fine on CPU.

---

## 4. API reference

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Frontend |
| POST | `/api/upload` | multipart `samples` field (multiple files) → preprocesses & stores them |
| POST | `/api/extract-style` | Runs CV + autoencoder on this session's samples → style profile JSON |
| POST | `/api/generate` | JSON `{text, font, page_size, font_size, ink_color, ruled_lines}` → generated page(s) |
| GET | `/api/download/<sid>/<file>` | Download a generated PNG/PDF |

Session state (uploaded samples, extracted profile) is kept **in memory**,
keyed by a session-id cookie Flask sets automatically.

---

## 5. Production notes

This is built to be a correct, complete local demo, not a hardened multi-user
service. Before deploying it beyond localhost:
- Replace the in-memory `SESSION_STORE` dict in `app.py` with Redis or a
  database so state survives restarts and works across multiple workers.
- Set a real `HANDWRITING_APP_SECRET` environment variable instead of the
  dev default in `app.py`.
- Add file-type/content validation beyond extension checking (e.g. Pillow's
  `Image.verify()`) before trusting uploaded files.
- Put it behind a proper WSGI server (gunicorn/uwsgi) instead of Flask's
  development server, and turn `debug=False` off in `app.py`.
- Add periodic cleanup of `uploads/` and `output/` — nothing currently
  expires old sessions' files.

---

## 6. Troubleshooting

- **`ModuleNotFoundError: No module named 'cv2'`** — you're not in the venv,
  or `pip install -r requirements.txt` didn't finish. Re-run it.
- **Torch install is slow/huge** — use the CPU-only wheel URL shown in step 3
  of Quick Start; it's a fraction of the size of the default CUDA build.
- **Uploaded sample looks blank/odd in the preview** — the Otsu binarization
  step assumes reasonably even lighting; try a flatter scan/photo, or a
  sample with more visible ink-to-paper contrast.
- **Port 5000 already in use** — change `app.run(port=5000)` in `app.py`'s
  last line, or stop whatever else is using that port.

---

## Future enhancements (from the original spec)
- Cursive-specific stroke joining
- Multiple saved handwriting profiles per user
- Cloud deployment
- Fine-tuning support once wired to a real neural backend
