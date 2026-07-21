"""
utils/export.py
-----------------
Saves rendered handwriting pages as PNG and multi-page PDF.
"""

from PIL import Image


def save_png(page: Image.Image, path: str):
    page.save(path, "PNG")


def save_pdf(pages: list, path: str):
    """Pillow can write multi-page PDFs directly from a list of RGB images."""
    if not pages:
        raise ValueError("No pages to export.")
    first, rest = pages[0], pages[1:]
    first = first.convert("RGB")
    rest = [p.convert("RGB") for p in rest]
    if rest:
        first.save(path, "PDF", resolution=150.0, save_all=True, append_images=rest)
    else:
        first.save(path, "PDF", resolution=150.0)
