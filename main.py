import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from src.detector import OllamaDetector

def main():
    parser = argparse.ArgumentParser(description="Ollama VLM Object Detection Prompt Engineering")
    parser.add_argument("--image", type=str, default="example_data/1_Color.png", help="Path to input image")
    parser.add_argument("--target", type=str, required=True, help="Description of the target object to detect")
    parser.add_argument("--prompt_template", type=str, default="Find the bounding box of {}", help="Template for the prompt, must contain {} placeholder.")
    parser.add_argument("--model", type=str, default="qwen3-vl:32b-instruct-q4_K_M", help="Ollama model name")
    parser.add_argument("--output", type=str, default="output_test.jpg", help="Path to save result image")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
        
    print(f"Initializing detector with model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Prompt Template: {args.prompt_template}")
    
    detector = OllamaDetector(model_name=args.model)
    
    # Run detection
    print("Sending prompt to model...")
    result = detector.detect(str(image_path), args.target, prompt_template=args.prompt_template)
    
    if not result['success']:
        print(f"Detection failed: {result.get('error')}")
        sys.exit(1)
        
    print("-" * 30)
    print("Raw Response:")
    print(result['raw_response'])
    print("-" * 30)
    
    coords_list = result.get('coordinates')
    if coords_list:
        print(f"Found {len(coords_list)} objects.")
        print(f"Normalized coordinates: {coords_list}")
        
        # Visualize
        try:
            img = Image.open(image_path)
            width, height = img.size
            draw = ImageDraw.Draw(img)
            
            for i, coords in enumerate(coords_list):
                # Qwen-VL defaults: [ymin, xmin, ymax, xmax] (0-1000 normalized)
                if len(coords) == 4:
                    y1, x1, y2, x2 = coords
                    
                    # Convert to pixels
                    px_x1 = (x1 / 1000) * width
                    px_y1 = (y1 / 1000) * height
                    px_x2 = (x2 / 1000) * width
                    px_y2 = (y2 / 1000) * height
                    
                    # Draw box
                    draw.rectangle([px_x1, px_y1, px_x2, px_y2], outline="red", width=3)
                    label = f"{args.target} {i+1}"
                    draw.text((px_x1, px_y1), label, fill="red")
            
            img.save(args.output)
            print(f"Visualize result saved to {args.output}")
            
        except Exception as e:
            print(f"Error visualizing: {e}")
    else:
        print("No coordinates parsed from the response.")
        print("Try tweaking the prompt or checking if the model supports bounding box output.")

if __name__ == "__main__":
    main()
