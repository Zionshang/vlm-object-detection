import sys
from pathlib import Path
from typing import Optional
# Removed PIL import

from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager
from src.parsers.bbox_parser import BoundingBoxParser
from src.core.detector import ObjectDetector
from src.utils.image_utils import draw_bounding_boxes

class StaticDetectionApp:
    def __init__(self, 
                 model_name: str, 
                 template_name: str = "standard_detection.v2",
                 prompts_dir: str = "prompts",
                 visualization_config: dict = None):

        self.model_name = model_name
        self.template_name = template_name
        self.prompts_dir = prompts_dir
        self.visualization_config = visualization_config or {}

        self._init_components()

    def _init_components(self):
        print(f"Initializing Static Detection...")
        
        self.llm_client = OllamaClient(model_name=self.model_name)
        self.prompt_manager = PromptManager(prompt_dir=self.prompts_dir)
        self.bbox_parser = BoundingBoxParser()
        self.detector = ObjectDetector(self.llm_client, self.bbox_parser)

    def run(self, image_path: str, target_object: str, output_path: str = None):
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image '{image_path}' not found.")
            return

        if not target_object:
             target_object = input("Enter target object description: ").strip()
             if not target_object:
                 return

        print(f"Processing image: {image_path}")
        print(f"Target: {target_object}")

        # Prepare Prompt
        try:
            prompt = self.prompt_manager.format_prompt(self.template_name, target=target_object)
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Run Detector (pass path directly)
        result = self.detector.run(str(image_path), prompt)

        # Output
        print("-" * 30)
        print("Raw Response:")
        print(result['raw_response'])
        coords = result.get('coordinates', [])
        
        if coords:
            print(f"Found {len(coords)} objects: {coords}")
            
            # Visualization
            if not output_path:
                output_path = str(image_path.parent / f"{image_path.stem}_result.jpg")
                
            box_color = self.visualization_config.get("box_color", "red")
            box_width = self.visualization_config.get("box_width", 3)
            
            labels = [f"{target_object} {i+1}" for i in range(len(coords))]
            draw_bounding_boxes(str(image_path), coords, output_path, labels, color=box_color, width=box_width)
        else:
            print("No objects detected.")

