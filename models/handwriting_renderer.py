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
    align: str = "left",
    ink_color: tuple = (30, 30, 60),
    show_ruled_lines: bool = True,
    seed: int = 42,
    char_paths: dict = None,
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
    if not style_profile:
        style_profile = {}
    slant_deg = style_profile.get("slant_deg", 0.0)
    slant_var = min(2.5, style_profile.get("stroke_width_var", 1.0)) * latent_modifiers["slant_var_mod"]
    baseline_jitter = min(3.5, style_profile.get("baseline_jitter", 1.5)) * latent_modifiers["jitter_mod"]
    letter_spacing_extra = min(10.0, style_profile.get("letter_spacing", 4.0)) * 0.15
    roughness = style_profile.get("roughness", 0.3)
    size_jitter = min(roughness, 0.3) * latent_modifiers["stroke_var_mod"]

    max_chars_per_line = int((W - 2 * margin_x) / (base_font_size * 0.55))
    lines = _wrap_text(text, max_chars_per_line)

    font = _load_font(font_key, base_font_size)

    # Preload char images if using exact template
    char_images = {}
    if char_paths:
        for ch, path in char_paths.items():
            if os.path.exists(path):
                char_images[ch] = Image.open(path).convert("RGBA")

    # Pre-calculate widths for alignment
    def get_char_advance(ch):
        if ch == " ":
            return base_font_size * 0.35 * latent_modifiers["flow_mod"]
        if char_images and ch in char_images:
            orig_img = char_images[ch]
            scale = (base_font_size * 1.5) / float(max(orig_img.height, 1))
            gw = max(2, int(orig_img.width * scale))
            return gw + letter_spacing_extra * latent_modifiers["flow_mod"]
        else:
            bbox = draw.textbbox((0, 0), ch, font=font)
            gw = bbox[2] - bbox[0]
            if gw <= 0: gw = int(base_font_size * 0.4)
            return gw + letter_spacing_extra * latent_modifiers["flow_mod"]

    cursor_y = margin_y
    for line in lines:
        if cursor_y + line_height > H - margin_y:
            break  # single-page render; caller can paginate for longer text
            
        line_margin_variance = rng.uniform(-15, 25)
        
        # Calculate alignment offset
        if align in ["center", "right"]:
            line_width = sum(get_char_advance(ch) for ch in line)
            max_width = W - 2 * margin_x
            if align == "center":
                cursor_x = margin_x + max(0, (max_width - line_width) / 2) + rng.uniform(-10, 10)
            elif align == "right":
                cursor_x = W - margin_x - line_width - rng.uniform(0, 15)
        else:
            # left alignment with human variance
            cursor_x = margin_x + line_margin_variance
        
        for ch in line:
            if ch == " ":
                # Human word spacing varies slightly
                space_width = base_font_size * rng.uniform(0.3, 0.45)
                cursor_x += space_width * latent_modifiers["flow_mod"]
                continue

            jitter_y = rng.uniform(-1, 1) * baseline_jitter
            jitter_size = 1.0 + rng.uniform(-0.03, 0.03) * size_jitter
            char_slant = slant_deg + rng.uniform(-1, 1) * slant_var * 0.3

            glyph_size = max(10, int(base_font_size * jitter_size))
            pad = 12

            if char_images and ch in char_images:
                # EXACT TEMPLATE PATH
                orig_img = char_images[ch]
                # Scale using a fixed factor so all characters retain their relative sizes.
                # Assuming the fixed height image corresponds to ~2x the base font size.
                scale = (base_font_size * 1.5) / float(max(orig_img.height, 1))
                new_w = max(2, int(orig_img.width * scale))
                new_h = max(2, int(orig_img.height * scale))
                
                # Apply ink color to the extracted alpha mask
                colored_img = Image.new("RGBA", orig_img.size)
                colored_draw = ImageDraw.Draw(colored_img)
                colored_draw.rectangle([0, 0, orig_img.width, orig_img.height], fill=ink_color+(255,))
                
                # Use the original image's alpha channel as mask
                final_char = Image.new("RGBA", orig_img.size, (0,0,0,0))
                final_char.paste(colored_img, (0,0), mask=orig_img.split()[3])
                
                scaled = final_char.resize((new_w, new_h), Image.LANCZOS)
                
                glyph_img = Image.new("RGBA", (new_w + pad * 2, new_h + pad * 2), (0, 0, 0, 0))
                glyph_img.paste(scaled, (pad, pad))
                
                gw, gh = new_w, new_h
                # Because we preserved the full cell height, the baseline is roughly at 70% of the image height.
                bbox_y_offset = int(gh * 0.70)
            else:
                # FALLBACK PROCEDURAL FONT PATH
                glyph_font = _load_font(font_key, glyph_size) if glyph_size != base_font_size else font
                bbox = draw.textbbox((0, 0), ch, font=glyph_font)
                gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if gw <= 0:
                    gw = int(base_font_size * 0.4)
                if gh <= 0:
                    gh = base_font_size

                glyph_img = Image.new("RGBA", (gw + pad * 2, gh + pad * 2), (0, 0, 0, 0))
                gdraw = ImageDraw.Draw(glyph_img)
                gdraw.text((pad - bbox[0], pad - bbox[1]), ch, font=glyph_font, fill=ink_color + (255,))
                bbox_y_offset = bbox[1]

            if abs(char_slant) > 0.1 and not char_images: # only slant generated fonts, not template
                cx, cy = pad + gw / 2.0, pad + gh / 2.0
                glyph_img = glyph_img.rotate(-char_slant, resample=Image.BICUBIC, center=(cx, cy))

            paste_x = int(cursor_x - pad)
            paste_y = int(cursor_y + jitter_y - pad + bbox_y_offset)
            page.paste(glyph_img, (paste_x, paste_y), glyph_img)

            advance = gw if char_images else (glyph_font.getlength(ch) if hasattr(glyph_font, "getlength") else gw)
            cursor_x += advance + letter_spacing_extra * latent_modifiers["flow_mod"]

        cursor_y += line_height

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
