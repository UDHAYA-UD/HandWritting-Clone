"""
config.py
----------
Central place for settings you might want to tweak.
"""

import os

# "builtin"      -> style_autoencoder.py + handwriting_renderer.py (default,
#                    works out of the box, no extra downloads)
# "deepwriting"  -> routes through models/neural_hooks.py, see README.md
# "diffusionpen" -> routes through models/neural_hooks.py, see README.md
HANDWRITING_BACKEND = os.environ.get("HANDWRITING_BACKEND", "builtin")

MAX_UPLOAD_MB = 25
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Preprocessing Debug Flag
DEBUG_PREPROCESSING = True
