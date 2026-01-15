import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

# Color map for common colors
COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}

def draw_on_image(img: np.ndarray, boxes: List[List[int]], labels: List[str] = None, 
                  color: str = "red", width: int = 3) -> np.ndarray:
    """
    Draw bounding boxes on a NumPy Image array (BGR format).
    Returns the modified image.
    """
    # Create copy to avoid modifying original array if needed, 
    # but here we modify in place if it's passed, or return modified.
    # To be safe and functional, we act on the passed object
    # but since python passes by reference, let's just modify it.
    
    img_height, img_width = img.shape[:2]
    
    # Get BGR color
    bgr_color = COLOR_MAP.get(color.lower(), (0, 0, 255))
    
    for i, coords in enumerate(boxes):
        if len(coords) == 4:
            y1, x1, y2, x2 = coords
            
            # Convert to pixels
            px_x1 = int((x1 / 1000) * img_width)
            px_y1 = int((y1 / 1000) * img_height)
            px_x2 = int((x2 / 1000) * img_width)
            px_y2 = int((y2 / 1000) * img_height)
            
            # Draw box
            cv2.rectangle(img, (px_x1, px_y1), (px_x2, px_y2), bgr_color, width)
            
            label_text = labels[i] if labels and i < len(labels) else f"Object {i+1}"
            
            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (px_x1, px_y1 - text_h - 5), (px_x1 + text_w, px_y1), bgr_color, -1)
            
            # Draw label
            cv2.putText(img, label_text, (px_x1, px_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return img

def draw_bounding_boxes(image_source: Union[str, Path, np.ndarray], boxes: List[List[int]], 
                        output_path: str = None, labels: List[str] = None, 
                        color: str = "red", width: int = 3):
    """
    Draw bounding boxes on the image.
    image_source: File path (str/Path) or NumPy Array (BGR).
    output_path: Path to save result.
    boxes: List of [y1, x1, y2, x2] normalized to 0-1000.
    """
    try:
        if isinstance(image_source, (str, Path)):
            img = cv2.imread(str(image_source))
            if img is None:
                print(f"Error: Could not read image from {image_source}")
                return False
        elif isinstance(image_source, np.ndarray):
            img = image_source.copy()
        else:
            print("Error: Unsupported image source type.")
            return False

        draw_on_image(img, boxes, labels, color, width)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Result saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error checking/drawing image: {e}")
        return False

