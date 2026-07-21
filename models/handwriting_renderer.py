"""
models/handwriting_renderer.py
--------------------------------
Style-conditioned handwriting page renderer.

Takes:
  - a style profile (classical CV metrics from utils/style_extraction.py,
    optionally nudged by the PyTorch latent modifiers from
    models/style_autoencoder.py)
  - freeform text
  - page/font options

...and renders a page image where every character is individually placed
with a slant, baseline offset, size jitter, and spacing derived from that
style profile, using a handwriting-style base font (glyph shapes) as the
starting point. This is what actually performs "handwriting synthesis" in
this build; see models/neural_hooks.py for wiring in a full neural
stroke-generation model (DeepWriting/DiffusionPen) instead.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "fonts")

FONT_CHOICES = {
    "neat": "IndieFlower-Regular.ttf",
    "flowing": "Caveat-Regular.ttf",
    "casual": "ShadowsIntoLight-Regular.ttf",
    "quick": "HomemadeApple-Regular.ttf",
}

PAGE_SIZES = {
    "a4": (1240, 1754),      # ~150 DPI
    "letter": (1275, 1650),
}


def _load_font(font_key: str, size: int) -> ImageFont.FreeTypeFont:
    filename = FONT_CHOICES.get(font_key, FONT_CHOICES["neat"])
    path = os.path.join(FONT_DIR, filename)
    return ImageFont.truetype(path, size)


def _wrap_text(text: str, max_chars_per_line: int) -> list:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for w in words:
            trial = (current + " " + w).strip()
            if len(trial) > max_chars_per_line and current:
                lines.append(current)
                current = w
            else:
                current = trial
        lines.append(current)
    return lines


def render_page(
    text: str,
    style_profile: dict,
    latent_modifiers: dict = None,
    font_key: str = "neat",
    page_size: str = "a4",
    base_font_size: int = 42,
    ink_color: tuple = (30, 30, 60),
    show_ruled_lines: bool = True,
    seed: int = 42,
) -> Image.Image:
    """Render one page of style-conditioned synthetic handwriting."""
    rng = random.Random(seed)
    latent_modifiers = latent_modifiers or {
        "jitter_mod": 1.0, "slant_var_mod": 1.0, "stroke_var_mod": 1.0, "flow_mod": 1.0
    }

    W, H = PAGE_SIZES.get(page_size, PAGE_SIZES["a4"])
    margin_x, margin_y = 90, 120
    line_height = int(base_font_size * 1.7)

    page = Image.new("RGB", (W, H), (250, 247, 238))
    draw = ImageDraw.Draw(page)

    if show_ruled_lines:
        y = margin_y
        while y < H - margin_y:
            draw.line([(margin_x - 20, y), (W - margin_x + 20, y)],
                      fill=(210, 205, 185), width=1)
            y += line_height

    # --- derive per-character rendering behavior from the style profile ---
    slant_deg = style_profile.get("slant_deg", 0.0)
    slant_var = style_profile.get("stroke_width_var", 1.0) * latent_modifiers["slant_var_mod"]
    baseline_jitter = style_profile.get("baseline_jitter", 2.0) * latent_modifiers["jitter_mod"]
    letter_spacing_extra = style_profile.get("letter_spacing", 6.0) * 0.35
    roughness = style_profile.get("roughness", 0.3)
    size_jitter = 1.0 + min(roughness, 0.6) * latent_modifiers["stroke_var_mod"]

    max_chars_per_line = int((W - 2 * margin_x) / (base_font_size * 0.62))
    lines = _wrap_text(text, max_chars_per_line)

    font = _load_font(font_key, base_font_size)

    cursor_y = margin_y - line_height + 20
    for line in lines:
        cursor_y += line_height
        if cursor_y > H - margin_y:
            break  # single-page render; caller can paginate for longer text
        cursor_x = margin_x
        for ch in line:
            if ch == " ":
                cursor_x += base_font_size * 0.45 * latent_modifiers["flow_mod"]
                continue

            jitter_y = rng.uniform(-1, 1) * baseline_jitter
            jitter_size = 1.0 + rng.uniform(-0.08, 0.08) * size_jitter
            char_slant = slant_deg + rng.uniform(-1, 1) * slant_var * 0.6

            glyph_size = max(10, int(base_font_size * jitter_size))
            glyph_font = _load_font(font_key, glyph_size) if glyph_size != base_font_size else font

            bbox = draw.textbbox((0, 0), ch, font=glyph_font)
            gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if gw <= 0:
                gw = int(base_font_size * 0.5)
            if gh <= 0:
                gh = base_font_size

            pad = 6
            glyph_img = Image.new("RGBA", (gw + pad * 2, gh + pad * 2), (0, 0, 0, 0))
            gdraw = ImageDraw.Draw(glyph_img)
            gdraw.text((pad - bbox[0], pad - bbox[1]), ch, font=glyph_font,
                       fill=ink_color + (255,))
            if abs(char_slant) > 0.1:
                glyph_img = glyph_img.rotate(char_slant, resample=Image.BICUBIC,
                                              expand=True)

            paste_y = int(cursor_y - gh + jitter_y)
            page.paste(glyph_img, (int(cursor_x - pad), paste_y), glyph_img)
            cursor_x += gw * 0.72 + letter_spacing_extra * latent_modifiers["flow_mod"]

    return page


def render_multi_page(text: str, style_profile: dict, latent_modifiers: dict = None,
                       **kwargs) -> list:
    """Splits long text across as many pages as needed and renders each one.
    Simple char-budget based pagination (good enough for a demo app; a
    production version would measure actual wrapped-line counts)."""
    chars_per_page = 1400
    chunks = [text[i:i + chars_per_page] for i in range(0, len(text), chars_per_page)] or [""]
    pages = []
    for idx, chunk in enumerate(chunks):
        pages.append(render_page(chunk, style_profile, latent_modifiers,
                                  seed=42 + idx, **kwargs))
    return pages
