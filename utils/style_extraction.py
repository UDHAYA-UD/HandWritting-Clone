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
    """Std-dev of intra-line vertical position wander."""
    if len(stats) < 2:
        return 2.0
    bottoms = stats[:, 1] + stats[:, 3]
    heights = stats[:, 3]
    avg_h = float(np.median(heights)) if len(heights) > 0 else 20.0

    # Cluster components into lines by Y proximity
    sorted_idx = np.argsort(bottoms)
    sorted_bottoms = bottoms[sorted_idx]

    lines = []
    current_line = [sorted_bottoms[0]]
    for b in sorted_bottoms[1:]:
        if abs(b - np.mean(current_line)) < max(15.0, avg_h * 0.8):
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]
    lines.append(current_line)

    jitters = [np.std(l) for l in lines if len(l) > 1]
    if not jitters:
        return 2.0
    return float(np.clip(np.mean(jitters), 0.5, 4.0))


def _letter_spacing(stats):
    """Average horizontal gap between neighboring connected components on the same line."""
    if len(stats) < 2:
        return 4.0
    # Group components by approximate Y line position first
    bottoms = stats[:, 1] + stats[:, 3]
    heights = stats[:, 3]
    avg_h = float(np.median(heights)) if len(heights) > 0 else 20.0

    sorted_idx = np.argsort(bottoms)
    lines_components = []
    current_line = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        b = bottoms[idx]
        current_b_mean = np.mean([bottoms[i] for i in current_line])
        if abs(b - current_b_mean) < max(15.0, avg_h * 0.8):
            current_line.append(idx)
        else:
            lines_components.append(current_line)
            current_line = [idx]
    lines_components.append(current_line)

    gaps = []
    for line_idx in lines_components:
        if len(line_idx) < 2:
            continue
        line_stats = stats[line_idx]
        ordered = line_stats[np.argsort(line_stats[:, 0])]
        for i in range(1, len(ordered)):
            prev_right = ordered[i - 1][0] + ordered[i - 1][2]
            cur_left = ordered[i][0]
            gap = cur_left - prev_right
            if 0 <= gap < 40:
                gaps.append(gap)
    if not gaps:
        return 4.0
    return float(np.clip(np.mean(gaps), 1.0, 15.0))



def _stroke_slant_deg(binary_ink: np.ndarray) -> float:
    """Estimate true stroke slant angle (forward/backward cursive lean) in degrees using image gradients."""
    gx = cv2.Sobel(binary_ink, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary_ink, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.hypot(gx, gy)
    mask = (mag > 40) & (binary_ink > 0)
    if not np.any(mask):
        return 0.0

    gy_valid = gy[mask]
    gx_valid = gx[mask]

    non_zero = np.abs(gy_valid) > 1e-3
    if not np.any(non_zero):
        return 0.0

    angles_rad = np.arctan(-gx_valid[non_zero] / gy_valid[non_zero])
    angles_deg = np.degrees(angles_rad)

    valid_slants = angles_deg[(angles_deg >= -40) & (angles_deg <= 40)]
    if valid_slants.size < 10:
        return 0.0

    return float(np.median(valid_slants))


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
    slant_deg = _stroke_slant_deg(binary_ink)

    metrics = {
        "slant_deg": slant_deg,
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
    keys = ["slant_deg", "stroke_width_px", "stroke_width_var", "baseline_jitter",
            "letter_spacing", "ink_density", "roughness"]
    profile = {}
    for k in keys:
        vals = [m[k] for m in per_sample_metrics]
        profile[k] = float(np.mean(vals))
    profile["n_samples"] = len(per_sample_metrics)

    # Recommend base handwriting font matching the user's slant and flow characteristics
    if abs(profile["slant_deg"]) > 4.0 or profile["roughness"] > 0.35:
        profile["recommended_font"] = "flowing"
    elif profile["roughness"] > 0.2:
        profile["recommended_font"] = "quick"
    else:
        profile["recommended_font"] = "neat"

    return profile

