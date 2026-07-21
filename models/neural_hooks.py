"""
models/neural_hooks.py
------------------------
Integration point for swapping the built-in style autoencoder (see
style_autoencoder.py) for a real pretrained handwriting-synthesis model such
as DeepWriting or DiffusionPen.

This file intentionally contains no downloaded checkpoints or vendored
research code - those projects ship their own large pretrained weights and
licenses that need to be fetched and reviewed directly from their authors.
What's here is the exact seam to wire one in. See README.md ->
"Using the real DeepWriting / DiffusionPen models" for the full walkthrough.

Contract expected by app.py:
    generate_strokes(style_reference_images: list[np.ndarray], text: str) -> Any
        Returns whatever downstream renderer you pair with the model (e.g. a
        list of (x, y, pen_state) points for a stroke-based model like
        DeepWriting, or a rendered image tensor for a diffusion model like
        DiffusionPen).

By default this raises NotImplementedError with setup instructions, and
app.py falls back to the procedural renderer (models/handwriting_renderer.py)
driven by utils/style_extraction.py + models/style_autoencoder.py.
"""

from typing import List
import numpy as np


class ExternalHandwritingModel:
    """Thin adapter class. Implement `load()` and `generate_strokes()` after
    following the setup steps in the README, then set
    HANDWRITING_BACKEND=deepwriting (or diffusionpen) in config.py / your
    environment to route generation through here instead of the built-in
    procedural renderer.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load(self):
        raise NotImplementedError(
            "Load your DeepWriting / DiffusionPen checkpoint here, e.g.:\n"
            "  import torch\n"
            "  self.model = torch.load(self.checkpoint_path, map_location='cpu')\n"
            "  self.model.eval()\n"
            "See README.md for where to obtain the checkpoint and repo."
        )

    def generate_strokes(self, style_reference_images: List[np.ndarray], text: str):
        raise NotImplementedError(
            "Call the loaded model's sampling/generation function here and "
            "return its output (stroke sequence or rendered image) for "
            "app.py to hand to the renderer / exporter."
        )
