"""
utils/preprocessing.py
----------------------
Classical computer-vision preprocessing pipeline for handwriting sample images.

Pipeline: load -> grayscale -> denoise -> binarize (Otsu) -> deskew -> crop to ink
content. Every step is done with OpenCV / NumPy; Pillow is used only for final
I/O so the result can be reused by both the style-extraction and preview code.

Every function returns plain NumPy arrays so it is easy to unit test or reuse
in a notebook independent of Flask.
"""

import cv2
import numpy as np
from PIL import Image


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image from disk as an OpenCV BGR array, upscaling tiny phone
    photos isn't needed but we cap the max dimension for speed."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {path}. Is it a valid image file?")

    max_dim = 1600
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light denoising that preserves stroke edges (bilateral filter)."""
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold -> ink pixels become 255 (white) on a black background,
    which is the convention the rest of the pipeline expects."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # remove tiny speckle noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh


def estimate_skew_angle(binary_ink: np.ndarray) -> float:
    """Estimate the dominant skew of handwriting using the minimum-area
    bounding rectangle of all ink pixels. Returns degrees, positive = CCW."""
    coords = np.column_stack(np.where(binary_ink > 0))
    if coords.shape[0] < 20:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # cv2.minAreaRect angle convention varies by OpenCV version; normalize to
    # the small range a handwritten line would realistically have.
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90
    return float(np.clip(angle, -20, 20))


def deskew(binary_ink: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return binary_ink
    (h, w) = binary_ink.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(binary_ink, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_to_content(binary_ink: np.ndarray, padding: int = 15) -> np.ndarray:
    coords = cv2.findNonZero(binary_ink)
    if coords is None:
        return binary_ink
    x, y, w, h = cv2.boundingRect(coords)
    y0 = max(0, y - padding)
    x0 = max(0, x - padding)
    y1 = min(binary_ink.shape[0], y + h + padding)
    x1 = min(binary_ink.shape[1], x + w + padding)
    return binary_ink[y0:y1, x0:x1]


def preprocess_sample(path: str) -> dict:
    """Runs the full pipeline on one uploaded sample.

    Returns a dict with:
      - binary: np.ndarray (uint8, 0/255) ink-on-black, deskewed & cropped
      - skew_angle: float degrees removed
      - preview: PIL.Image (ink-on-white) suitable for saving/display
    """
    bgr = load_image_bgr(path)
    gray = to_grayscale(bgr)
    gray = denoise(gray)
    binary = binarize(gray)
    angle = estimate_skew_angle(binary)
    binary = deskew(binary, angle)
    binary = crop_to_content(binary)

    # Build a human-viewable preview: ink-on-white
    preview_arr = 255 - binary
    preview = Image.fromarray(preview_arr).convert("L")

    return {"binary": binary, "skew_angle": angle, "preview": preview}
