"""
app.py
------
Flask backend for the AI Handwriting Style Generator.

Workflow implemented (matches the project spec):
  upload handwriting samples -> preprocess -> extract style -> enter custom
  text -> generate handwriting -> preview -> download PNG/PDF

Endpoints
---------
GET  /                          Serves the single-page frontend.
POST /api/upload                Accepts multiple sample images, preprocesses
                                 each, stores them under a session, returns
                                 before/after preview URLs.
POST /api/extract-style         Runs style-metric extraction + trains the
                                 small PyTorch style autoencoder on this
                                 session's samples. Returns the style profile.
POST /api/generate              Renders page(s) of handwriting for typed
                                 text using the session's style profile.
GET  /api/download/<file_id>.png
GET  /api/download/<file_id>.pdf
GET  /uploads/<path:filename>   Serves stored sample previews.
GET  /output/<path:filename>    Serves generated pages.

Session state (uploaded sample paths, extracted style profile) is kept
in-memory keyed by a session id cookie. That's intentional for a local
single-user demo; see README.md's "Production notes" for what to change
(e.g. Redis-backed sessions) to run this for multiple concurrent users.
"""

import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, session

from utils.preprocessing import preprocess_sample
from utils.style_extraction import analyze_sample, aggregate_style_profile
from models.style_autoencoder import train_style_encoder, latent_to_style_modifiers
from models.handwriting_renderer import render_multi_page, FONT_CHOICES
from utils.export import save_png, save_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("HANDWRITING_APP_SECRET", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB total upload cap

# In-memory session store: {session_id: {"samples": [...], "style_profile": {...}}}
SESSION_STORE = {}


def get_session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    SESSION_STORE.setdefault(sid, {"samples": [], "style_profile": None, "latent_mods": None})
    return sid


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html", font_choices=list(FONT_CHOICES.keys()))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    sid = get_session_id()
    files = request.files.getlist("samples")
    if not files:
        return jsonify({"error": "No files received. Field name must be 'samples'."}), 400

    sample_dir = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(sample_dir, exist_ok=True)

    results = []
    for f in files:
        if not f.filename or not allowed_file(f.filename):
            continue
        sample_id = uuid.uuid4().hex[:10]
        ext = f.filename.rsplit(".", 1)[1].lower()
        raw_path = os.path.join(sample_dir, f"{sample_id}_original.{ext}")
        f.save(raw_path)

        try:
            processed = preprocess_sample(raw_path)
        except ValueError as e:
            os.remove(raw_path)
            return jsonify({"error": str(e)}), 400

        preview_name = f"{sample_id}_preview.png"
        preview_path = os.path.join(sample_dir, preview_name)
        processed["preview"].save(preview_path)

        entry = {
            "id": sample_id,
            "raw_path": raw_path,
            "preview_path": preview_path,
            "skew_angle": processed["skew_angle"],
        }
        SESSION_STORE[sid]["samples"].append(entry)
        results.append({
            "id": sample_id,
            "original_url": f"/uploads/{sid}/{os.path.basename(raw_path)}",
            "preview_url": f"/uploads/{sid}/{preview_name}",
            "skew_angle_deg": round(processed["skew_angle"], 2),
        })

    if not results:
        return jsonify({"error": "No valid image files found. Use PNG/JPG/BMP/WEBP."}), 400

    return jsonify({"samples": results, "total_samples": len(SESSION_STORE[sid]["samples"])})


@app.route("/api/extract-style", methods=["POST"])
def api_extract_style():
    sid = get_session_id()
    samples = SESSION_STORE[sid]["samples"]
    if not samples:
        return jsonify({"error": "Upload at least one handwriting sample first."}), 400

    from utils.preprocessing import load_image_bgr, to_grayscale, denoise, binarize, deskew, crop_to_content

    per_sample_metrics = []
    all_patches = []
    skew_angles = []
    for s in samples:
        # Re-run the deterministic preprocessing to get the binary ink mask
        # (kept cheap; re-uses the same pipeline as upload time).
        bgr = load_image_bgr(s["raw_path"])
        gray = denoise(to_grayscale(bgr))
        binary = binarize(gray)
        binary = deskew(binary, s["skew_angle"])
        binary = crop_to_content(binary)

        metrics, patches = analyze_sample(binary)
        per_sample_metrics.append(metrics)
        all_patches.append(patches)
        skew_angles.append(s["skew_angle"])

    style_profile = aggregate_style_profile(per_sample_metrics)
    style_profile["skew_angle_deg"] = float(np.mean(skew_angles))

    patch_stack = np.concatenate(all_patches, axis=0)
    latent = train_style_encoder(patch_stack)
    latent_mods = latent_to_style_modifiers(latent)

    SESSION_STORE[sid]["style_profile"] = style_profile
    SESSION_STORE[sid]["latent_mods"] = latent_mods

    return jsonify({"style_profile": style_profile, "latent_modifiers": latent_mods})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    sid = get_session_id()
    state = SESSION_STORE[sid]
    if not state["style_profile"]:
        return jsonify({"error": "Extract a style profile before generating."}), 400

    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to generate."}), 400

    font_key = data.get("font", "neat")
    page_size = data.get("page_size", "a4")
    font_size = int(data.get("font_size", 42))
    ink_color = tuple(data.get("ink_color", [30, 30, 60]))
    ruled_lines = bool(data.get("ruled_lines", True))

    pages = render_multi_page(
        text,
        state["style_profile"],
        state["latent_mods"],
        font_key=font_key,
        page_size=page_size,
        base_font_size=font_size,
        ink_color=ink_color,
        show_ruled_lines=ruled_lines,
    )

    file_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(OUTPUT_DIR, sid)
    os.makedirs(out_dir, exist_ok=True)

    png_paths = []
    for i, page in enumerate(pages):
        png_name = f"{file_id}_p{i + 1}.png"
        png_path = os.path.join(out_dir, png_name)
        save_png(page, png_path)
        png_paths.append(png_name)

    pdf_name = f"{file_id}.pdf"
    save_pdf(pages, os.path.join(out_dir, pdf_name))

    return jsonify({
        "file_id": file_id,
        "page_previews": [f"/output/{sid}/{n}" for n in png_paths],
        "png_urls": [f"/api/download/{sid}/{n}" for n in png_paths],
        "pdf_url": f"/api/download/{sid}/{pdf_name}",
    })


@app.route("/uploads/<sid>/<path:filename>")
def serve_upload(sid, filename):
    return send_from_directory(os.path.join(UPLOAD_DIR, sid), filename)


@app.route("/output/<sid>/<path:filename>")
def serve_output_preview(sid, filename):
    return send_from_directory(os.path.join(OUTPUT_DIR, sid), filename)


@app.route("/api/download/<sid>/<path:filename>")
def download_output(sid, filename):
    return send_from_directory(os.path.join(OUTPUT_DIR, sid), filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)

