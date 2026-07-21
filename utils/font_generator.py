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

TEMPLATE_MASK = None

def get_template_mask():
    global TEMPLATE_MASK
    if TEMPLATE_MASK is not None:
        return TEMPLATE_MASK
    
    tmp_path = "blank_template.png"
    if not os.path.exists(tmp_path):
        generate_template(tmp_path)
        
    img = cv2.imread(tmp_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # dilate significantly to provide a margin of error for minor paper warping/homography inaccuracies
    TEMPLATE_MASK = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return TEMPLATE_MASK


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect


def process_template(image_path, output_dir):
    """
    Reads the filled template, extracts characters, saves them to output_dir.
    Returns a dict { 'char': 'path/to/char.png' }
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read template image.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_coarse = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    
    contours, _ = cv2.findContours(thresh_coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find any contours in the image.")
        
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(largest)
        pts = cv2.boxPoints(rect)
        
    pts = order_points(pts)
    dst_pts = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (W, H))
    
    # 2. Illumination flatten
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    from utils.preprocessing import flatten_illumination
    warped_gray = flatten_illumination(warped_gray)
    
    # 3. Adaptive threshold
    blur2 = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 25)
    
    # 1. Apply static template mask to wipe out binarized grid lines and labels
    t_mask = get_template_mask()
    thresh[t_mask > 0] = 0
    
    # 4. Morphological open to remove speckles
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cell_w = (W - 2 * MARGIN_X) // COLS
    cell_h = (H - 2 * MARGIN_Y) // ROWS
    
    char_paths = {}
    
    for idx, char in enumerate(CHARS):
        r = idx // COLS
        c = idx % COLS
        x0 = MARGIN_X + c * cell_w
        y0 = MARGIN_Y + r * cell_h
        
        # Crop slightly smaller to avoid any remaining edges of the grid cell
        crop_pad = 5
        cell_img = thresh[y0+crop_pad:y0+cell_h-crop_pad, x0+crop_pad:x0+cell_w-crop_pad].copy()
        
        # Close gaps to reconnect broken grid lines caused by uneven lighting/thresholding
        closed_vert = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8))
        closed_horiz = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
        closed = cv2.bitwise_or(closed_vert, closed_horiz)
        
        # Remove any grid lines that bled into the crop by checking connected components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        for i in range(1, n):
            h_stat = stats[i, cv2.CC_STAT_HEIGHT]
            w_stat = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # If a component spans almost the entire height or width of the cell, it's a grid line
            is_grid_line = (h_stat >= cell_img.shape[0] - 10) or (w_stat >= cell_img.shape[1] - 10)
            
            # If a component is tiny, it's noise
            is_noise = area < 30
            
            if is_grid_line or is_noise:
                cell_img[labels == i] = 0
                
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
