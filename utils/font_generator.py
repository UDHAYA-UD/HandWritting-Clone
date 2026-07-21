import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-"
COLS = 7
ROWS = 10
W, H = 1240, 1754
MARGIN_X, MARGIN_Y = 50, 100

def get_layout_path():
    return os.path.join(os.path.dirname(__file__), "template_v2_layout.json")

def generate_template(output_path):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    cell_w = (W - 2 * MARGIN_X) // COLS
    cell_h = (H - 2 * MARGIN_Y) // ROWS
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        
    layout = {
        "page_size": [W, H],
        "fiducials": {},
        "cells": {}
    }
    
    # 1. Draw fiducials (4 corner squares ~26x26px)
    fid_size = 26
    fid_margin = 20
    
    fids = {
        "top_left": [fid_margin, fid_margin, fid_margin+fid_size, fid_margin+fid_size],
        "top_right": [W-fid_margin-fid_size, fid_margin, W-fid_margin, fid_margin+fid_size],
        "bottom_left": [fid_margin, H-fid_margin-fid_size, fid_margin+fid_size, H-fid_margin],
        "bottom_right": [W-fid_margin-fid_size, H-fid_margin-fid_size, W-fid_margin, H-fid_margin]
    }
    
    for key, box in fids.items():
        draw.rectangle(box, fill=(0, 0, 0))
        layout["fiducials"][key] = box
        
    # 2. Draw cells
    for idx, char in enumerate(CHARS):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN_X + c * cell_w
        y0 = MARGIN_Y + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        
        # Don't draw the grid!
        
        # Visually match the provided template:
        # Label is at the top-left-ish, above the writing baseline
        label_text = char
        try:
            bbox = draw.textbbox((0,0), label_text, font=font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
        except:
            lw, lh = 10, 15
            
        lx = x0 + 60
        ly = y0 + 20
        draw.text((lx, ly), label_text, fill=(160, 160, 160), font=font)
        
        # Dot is on the far-left, marking the baseline (below the label, leaving room for descending tails below it)
        dx = x0 + 20
        dy = y0 + 80
        dot_r = 4
        draw.ellipse([dx-dot_r, dy-dot_r, dx+dot_r, dy+dot_r], fill=(0, 0, 0))
        
        # Save metadata
        pad = 5
        layout["cells"][char] = {
            "cell_crop_box": [x0, y0, x1, y1],
            "label_box": [lx-pad, ly-pad, lx+lw+pad, ly+lh+pad],
            "dot_box": [dx-dot_r-pad, dy-dot_r-pad, dx+dot_r+pad, dy+dot_r+pad],
            "dot_center": [dx, dy]
        }
        
    img.save(output_path)
    
    with open(get_layout_path(), "w") as f:
        json.dump(layout, f, indent=2)
        
    return output_path

def find_fiducials(gray_img):
    _, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        # Fiducial is ~26x26 on 1240x1754, so depending on camera res it might vary.
        # We'll allow a wide range. 300 to 5000 pixels.
        if 200 < area < 5000 and 0.5 < aspect < 2.0:
            # Check solidity to ensure it's a filled shape
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area > 0.8:
                # Must be near the corners (within 30% of image edges)
                img_h, img_w = gray_img.shape[:2]
                cx, cy = x + w // 2, y + h // 2
                near_edge_x = cx < img_w * 0.3 or cx > img_w * 0.7
                near_edge_y = cy < img_h * 0.3 or cy > img_h * 0.7
                if near_edge_x and near_edge_y:
                    candidates.append([cx, cy])
    
    if len(candidates) != 4:
        raise ValueError(f"Couldn't detect exactly 4 corner markers (found {len(candidates)}). Retake the photo with all 4 corners visible and well-lit.")
        
    # Sort candidates (TL, TR, BR, BL)
    pts = np.array(candidates, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)] # TL
    ordered[2] = pts[np.argmax(s)] # BR
    ordered[1] = pts[np.argmin(diff)] # TR
    ordered[3] = pts[np.argmax(diff)] # BL
    
    return ordered

def process_template(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read template image.")
        
    layout_path = get_layout_path()
    if not os.path.exists(layout_path):
        # We must generate the template at least once to create the JSON
        generate_template(os.path.join(output_dir, "temp_handwriting_template.png"))
        
    with open(layout_path, "r") as f:
        layout = json.load(f)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    src_pts = find_fiducials(gray)
    
    fids = layout["fiducials"]
    # Order: TL, TR, BR, BL
    dst_pts = np.array([
        [fids["top_left"][0] + 13, fids["top_left"][1] + 13], # center of TL box
        [fids["top_right"][0] + 13, fids["top_right"][1] + 13], # center of TR box
        [fids["bottom_right"][0] + 13, fids["bottom_right"][1] + 13], # center of BR box
        [fids["bottom_left"][0] + 13, fids["bottom_left"][1] + 13]  # center of BL box
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, tuple(layout["page_size"]))
    
    # Save debug warped image if requested
    if os.environ.get("DEBUG_PREPROCESSING") == "True":
        cv2.imwrite(os.path.join(output_dir, "debug_warped.png"), warped)
        
    # Unconditional Whiteout of labels and dots
    for cell in layout["cells"].values():
        for box_key in ("label_box", "dot_box"):
            x0, y0, x1, y1 = [int(v) for v in cell[box_key]]
            warped[y0:y1, x0:x1] = [255, 255, 255]
            
    # Flatten Illumination & Adaptive Threshold
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    from utils.preprocessing import flatten_illumination
    warped_gray = flatten_illumination(warped_gray)
    
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 25)
    
    # Open to remove tiny noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    
    if os.environ.get("DEBUG_PREPROCESSING") == "True":
        cv2.imwrite(os.path.join(output_dir, "debug_masked.png"), binary)
        
    char_paths = {}
    
    for char, cell in layout["cells"].items():
        x0, y0, x1, y1 = [int(v) for v in cell["cell_crop_box"]]
        
        # Directly crop using metadata bounds, padded slightly inward to avoid page border artifacts
        crop_pad = 10
        cell_img = binary[y0+crop_pad:y1-crop_pad, x0+crop_pad:x1-crop_pad]
        
        coords = cv2.findNonZero(cell_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w > 1 and h > 1:
                ink_crop = cell_img[:, x:x+w]
                crop_h = ink_crop.shape[0]
                mask = Image.fromarray(ink_crop)
                
                rgba = Image.new("RGBA", (w, crop_h), (0, 0, 0, 0))
                rgba_data = []
                for p in mask.getdata():
                    rgba_data.append((0, 0, 0, p))
                rgba.putdata(rgba_data)
                
                safe_char = "dot" if char == "." else \
                            "comma" if char == "," else \
                            "exc" if char == "!" else \
                            "que" if char == "?" else \
                            "apos" if char == "'" else \
                            "quot" if char == '"' else \
                            "dash" if char == "-" else char
                            
                if safe_char.isupper():
                    safe_char = safe_char + "_upper"
                    
                path = os.path.join(output_dir, f"{safe_char}.png")
                rgba.save(path)
                char_paths[char] = path
                
    return char_paths
