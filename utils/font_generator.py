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


def ink_color_mask(bgr_img):
    """Filters out neutral-grey printed labels by keeping only saturated pixels (blue ink)."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    # tune threshold between 20-50 based on pen color
    _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
    return sat_mask


def crop_label_region(cell_img, label_frac=0.22):
    """Wipes out the top-left corner where the printed label sits."""
    h, w = cell_img.shape[:2]
    # Set to 0 because our mask uses 255 for ink, 0 for background
    cell_img[:int(h * label_frac), :int(w * label_frac)] = 0
    return cell_img


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
    
    # Blur to reduce speckle noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold with larger block size (51) and C (25) to handle harsh lighting/shadows
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 25)
                                   
    # Remove remaining tiny speckles
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Generate the color saturation mask from the original resized image
    sat_mask = ink_color_mask(img)
                                   
    cell_w = (W - 2 * MARGIN_X) // COLS
    cell_h = (H - 2 * MARGIN_Y) // ROWS
    
    char_paths = {}
    
    for idx, char in enumerate(CHARS):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN_X + c * cell_w
        y0 = MARGIN_Y + r * cell_h
        
        # Crop slightly smaller than the full cell to avoid adjacent cell ink, but keep the full height and label area
        crop_top = 5
        crop_bottom = 5
        crop_x = 5
        cell_img = thresh[y0+crop_top:y0+cell_h-crop_bottom, x0+crop_x:x0+cell_w-crop_x].copy()
        cell_sat = sat_mask[y0+crop_top:y0+cell_h-crop_bottom, x0+crop_x:x0+cell_w-crop_x].copy()
        
        # 1. Fallback: Forcefully wipe out the top-left label region
        cell_img = crop_label_region(cell_img)
        
        # 2. Color filter: Only keep pixels that are both dark (thresh) AND saturated (blue ink)
        cell_img = cv2.bitwise_and(cell_img, cell_sat)
        
        # Close gaps to reconnect broken grid lines caused by uneven lighting/thresholding
        closed_vert = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8))
        closed_horiz = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
        closed = cv2.bitwise_or(closed_vert, closed_horiz)
        
        # Remove any grid lines that bled into the crop by checking connected components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        # We will clear noise on the ORIGINAL cell_img, using the closed CC stats
        for i in range(1, n):
            h_stat = stats[i, cv2.CC_STAT_HEIGHT]
            w_stat = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            # If a component spans almost the entire height or width of the cell, it's a grid line
            is_grid_line = (h_stat >= cell_img.shape[0] - 10) or (w_stat >= cell_img.shape[1] - 10)
            
            # If a component is in the top-left and small, it's the printed reference label (a, b, c...)
            is_label = (cx < 60 and cy < 50 and area < 800)
            
            # If a component is tiny, it's noise
            is_noise = area < 30
            
            if is_grid_line or is_label or is_noise:
                cell_img[labels == i] = 0
        
        # Find ink bounding box
        coords = cv2.findNonZero(cell_img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w > 1 and h > 1:
                # Keep full vertical height of the cell to preserve baseline offset
                ink_crop = cell_img[:, x:x+w]
                crop_h = ink_crop.shape[0]
                mask = Image.fromarray(ink_crop)
                
                rgba = Image.new("RGBA", (w, crop_h), (0, 0, 0, 0))
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
