from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Tuple

def draw_bounding_boxes(image_path: str, boxes: List[List[int]], output_path: str, labels: List[str] = None, 
                        color: str = "red", width: int = 3):
    """
    Draw bounding boxes on the image.
    boxes: List of [y1, x1, y2, x2] normalized to 0-1000.
    """
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)
        
        for i, coords in enumerate(boxes):
            if len(coords) == 4:
                y1, x1, y2, x2 = coords
                
                # Convert to pixels
                px_x1 = (x1 / 1000) * img_width
                px_y1 = (y1 / 1000) * img_height
                px_x2 = (x2 / 1000) * img_width
                px_y2 = (y2 / 1000) * img_height
                
                # Draw box
                draw.rectangle([px_x1, px_y1, px_x2, px_y2], outline=color, width=width)
                
                label_text = labels[i] if labels and i < len(labels) else f"Object {i+1}"
                draw.text((px_x1, px_y1), label_text, fill=color)
        
        img.save(output_path)
        print(f"Result saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error checking/drawing image: {e}")
        return False
