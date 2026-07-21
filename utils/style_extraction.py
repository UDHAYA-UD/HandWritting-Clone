"""
utils/style_extraction.py
--------------------------
Turns preprocessed (binarized, deskewed) handwriting samples into a
quantitative "style profile": a small set of numbers that describe how a
person writes. These numbers directly drive the procedural renderer in
models/handwriting_renderer.py, and are also handed to the PyTorch style
autoencoder (models/style_autoencoder.py) as normalized image patches so the
learned latent code can modulate the same parameters with texture the
hand-coded metrics can't capture.

Metrics extracted, per sample, then averaged across all uploaded samples:
  - slant_deg          average slant of strokes (cursive lean)
  - stroke_width_px    average ink stroke thickness
  - stroke_width_var   variability of stroke thickness (pen pressure)
  - letter_spacing     average horizontal gap between connected components
  - baseline_jitter    vertical wobble of each glyph's baseline
  - ink_density        fraction of the page that is ink (roughly "heaviness")
  - roughness          contour irregularity (0 = smooth print, 1 = jagged/fast writing)
"""

import cv2
import numpy as np


def _connected_components(binary_ink: np.ndarray):
    """Return stats for connected ink components (roughly: strokes/glyphs)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_ink, connectivity=8)
    # skip label 0 (background)
    return n - 1, labels, stats[1:], centroids[1:]


def _stroke_width_stats(binary_ink: np.ndarray):
    """Distance-transform based estimate of stroke thickness: for every ink
    pixel, the distance transform gives roughly half the local stroke width."""
    dist = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
    ink_dist = dist[binary_ink > 0]
    if ink_dist.size == 0:
        return 2.0, 0.5
    widths = ink_dist * 2.0
    return float(np.mean(widths)), float(np.std(widths))


def _baseline_jitter(stats, centroids, img_h):
    """Std-dev of the vertical position (bottom edge) of each connected
    component - a proxy for how much a person's writing wanders off the line."""
    if len(stats) < 2:
        return 2.0
    bottoms = stats[:, 1] + stats[:, 3]  # y + height
    return float(np.clip(np.std(bottoms), 0.5, img_h * 0.15))


def _letter_spacing(stats):
    """Average horizontal gap between neighboring connected components on
    (roughly) the same text line."""
    if len(stats) < 2:
        return 6.0
    # sort left-to-right
    ordered = stats[np.argsort(stats[:, 0])]
    gaps = []
    for i in range(1, len(ordered)):
        prev_right = ordered[i - 1][0] + ordered[i - 1][2]
        cur_left = ordered[i][0]
        gap = cur_left - prev_right
        if 0 <= gap < 60:  # ignore huge gaps = new line / word boundary outliers
            gaps.append(gap)
    if not gaps:
        return 6.0
    return float(np.clip(np.mean(gaps), 1.0, 40.0))


def _roughness(binary_ink: np.ndarray):
    """Ratio of contour perimeter to convex-hull perimeter, averaged over
    components. Higher = jagged / fast handwriting, lower = smooth print."""
    contours, _ = cv2.findContours(binary_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ratios = []
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        perim = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        hull_perim = cv2.arcLength(hull, True)
        if hull_perim > 0:
            ratios.append(perim / hull_perim)
    if not ratios:
        return 0.3
    val = (np.mean(ratios) - 1.0)  # 1.0 = perfectly convex/smooth
    return float(np.clip(val, 0.0, 1.0))


def extract_patches(binary_ink: np.ndarray, stats, patch_size=64, max_patches=24):
    """Crop individual glyph/stroke patches (resized to patch_size^2) to feed
    the PyTorch style autoencoder. Returns an (N, patch_size, patch_size)
    float32 array normalized to [0, 1]."""
    patches = []
    # largest components first - these are more likely to be full letters
    # rather than punctuation / noise specks
    ordered = stats[np.argsort(-stats[:, 4])]  # sort by area desc
    for s in ordered[:max_patches]:
        x, y, w, h, _area = s
        if w < 4 or h < 4:
            continue
        crop = binary_ink[y:y + h, x:x + w]
        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        patches.append(crop.astype(np.float32) / 255.0)
    if not patches:
        patches = [np.zeros((patch_size, patch_size), dtype=np.float32)]
    return np.stack(patches, axis=0)


def analyze_sample(binary_ink: np.ndarray) -> dict:
    """Compute the full style-metric dict for a single preprocessed sample."""
    n_components, _labels, stats, centroids = _connected_components(binary_ink)
    stroke_w, stroke_w_var = _stroke_width_stats(binary_ink)
    ink_density = float(np.count_nonzero(binary_ink)) / float(binary_ink.size)

    metrics = {
        "stroke_width_px": stroke_w,
        "stroke_width_var": stroke_w_var,
        "baseline_jitter": _baseline_jitter(stats, centroids, binary_ink.shape[0]) if n_components else 2.0,
        "letter_spacing": _letter_spacing(stats) if n_components else 6.0,
        "ink_density": ink_density,
        "roughness": _roughness(binary_ink),
        "n_components": n_components,
    }
    patches = extract_patches(binary_ink, stats) if n_components else \
        np.zeros((1, 64, 64), dtype=np.float32)
    return metrics, patches


def aggregate_style_profile(per_sample_metrics: list) -> dict:
    """Average metrics across every uploaded sample into one style profile."""
    keys = ["stroke_width_px", "stroke_width_var", "baseline_jitter",
            "letter_spacing", "ink_density", "roughness"]
    profile = {}
    for k in keys:
        vals = [m[k] for m in per_sample_metrics]
        profile[k] = float(np.mean(vals))
    profile["n_samples"] = len(per_sample_metrics)
    return profile
