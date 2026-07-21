import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-"
COLS = 7
ROWS = 10
W, H = 1240, 1754
MARGIN_X, MARGIN_Y = 50, 100

def generate_template(output_path):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    cell_w = (W - 2 * MARGIN_X) // COLS
    cell_h = (H - 2 * MARGIN_Y) // ROWS
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        
    for idx, char in enumerate(CHARS):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN_X + c * cell_w
        y0 = MARGIN_Y + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        
        # Draw box
        draw.rectangle([x0, y0, x1, y1], outline=(200, 200, 200), width=2)
        # Draw label
        draw.text((x0 + 10, y0 + 10), char, fill=(180, 180, 180), font=font)
        
    img.save(output_path)
    return output_path

def process_template(image_path, output_dir):
    """
    Reads the filled template, extracts characters, saves them to output_dir.
    Returns a dict { 'char': 'path/to/char.png' }
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read template image.")
        
    # Resize to exact dimensions (assumes user uploaded a full-page photo of the template)
    img = cv2.resize(img, (W, H))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to handle lighting variations
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 15)
                                   
    cell_w = (W - 2 * MARGIN_X) // COLS
    cell_h = (H - 2 * MARGIN_Y) // ROWS
    
    char_paths = {}
    
    for idx, char in enumerate(CHARS):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN_X + c * cell_w
        y0 = MARGIN_Y + r * cell_h
        
        # Crop with padding to avoid box lines and label text
        pad_top = 45 # avoid the printed label at top left
        pad = 10
        cell_img = thresh[y0+pad_top:y0+cell_h-pad, x0+pad:x0+cell_w-pad]
        
        # Find ink bounding box
        coords = cv2.findNonZero(cell_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w > 5 and h > 5:
                ink_crop = cell_img[y:y+h, x:x+w]
                mask = Image.fromarray(ink_crop)
                
                rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                # Put pure black ink with the mask as alpha
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
                            
                # Handle case sensitivity in windows file paths
                if safe_char.isupper():
                    safe_char = safe_char + "_upper"
                    
                path = os.path.join(output_dir, f"{safe_char}.png")
                rgba.save(path)
                char_paths[char] = path
                
    return char_paths
