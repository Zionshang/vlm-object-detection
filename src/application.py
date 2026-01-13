import cv2
import time
from pathlib import Path
from PIL import Image
import sys

from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager
from src.parsers.bbox_parser import BoundingBoxParser
from src.pipeline import DetectionPipeline
from src.utils.image_utils import draw_bounding_boxes
from src.utils.logger import ExperimentLogger
from src.utils.camera import RealSenseCamera

class LiveDetectionApp:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self._init_components()

    def _init_components(self):
        print(f"Initializing Experiment: {self.args.experiment_name}")
        print(f"Model: {self.args.model}")
        
        self.llm_client = OllamaClient(model_name=self.args.model)
        self.prompt_manager = PromptManager(prompt_dir=self.config.get("prompts_dir", "prompts"))
        self.bbox_parser = BoundingBoxParser()
        self.pipeline = DetectionPipeline(self.llm_client, self.bbox_parser)
        self.logger = ExperimentLogger(log_file=self.config.get("log_file", "logs/experiments.jsonl"))

    def run_detection(self, inference_path, target_object, visualization_tasks=None):
        """
        Run detection and visualize results on multiple images.
        visualization_tasks: List of (source_image_path, save_result_path) pairs.
        """
        if visualization_tasks is None:
            visualization_tasks = []

        # Prepare Prompt
        try:
            prompt = self.prompt_manager.format_prompt(self.args.template, target=target_object)
            print(f"Constructed Prompt: {prompt}")
        except ValueError as e:
            print(f"Error constructing prompt: {e}")
            return

        # Execute Pipeline
        print("Running pipeline...")
        result = self.pipeline.run(inference_path, prompt)

        # Log
        result['experiment_name'] = self.args.experiment_name
        result['model'] = self.args.model
        result['template_key'] = self.args.template
        result['target'] = target_object
        result['image_path'] = inference_path
        
        self.logger.log(result)
        
        print("-" * 30)
        print("Raw Response:")
        print(result['raw_response'])
        print("-" * 30)
        
        if self.config.get("show_latency", False):
            print(f"Inference Latency: {result.get('latency_seconds', 0):.4f}s")
            print("-" * 30)
        
        coords = result.get('coordinates', [])
        if coords:
            print(f"Found {len(coords)} objects.")
            labels = [f"{target_object} {i+1}" for i in range(len(coords))]
            
            # Use visualization settings
            vis_config = self.config.get("visualization", {})
            box_color = vis_config.get("box_color", "red")
            box_width = vis_config.get("box_width", 3)
            
            # Visualize on all requested images
            for source_path, save_path in visualization_tasks:
                draw_bounding_boxes(source_path, coords, save_path, labels, color=box_color, width=box_width)
        else:
            print("No objects detected or parsing failed.")

    def run(self):
        print("Starting Live Mode... (Press SPACE to capture, 'q' to quit)")
        
        # Initialize camera with config
        cam_config = self.config.get("camera", {})
        camera = RealSenseCamera(
            width=cam_config.get("width", 1280),
            height=cam_config.get("height", 720),
            fps=cam_config.get("fps", 30),
            warmup=cam_config.get("warmup_seconds", 1.0)
        )
        if not camera.pipeline:
             print("Failed to start camera. Exiting.")
             return

        cv2.namedWindow("RealSense Feed", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # 1. Get continuous frame
                frame_rgb = camera.get_frame()
                if frame_rgb is None:
                    continue
                
                # Convert for OpenCV display (RGB -> BGR)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.putText(frame_bgr, "Live Preview - Press SPACE to Detect", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RealSense Feed", frame_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 32: # SPACE
                    self._handle_capture(camera, frame_rgb, frame_bgr)

        except KeyboardInterrupt:
            pass
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print("Exiting Live Mode.")

    def _handle_capture(self, camera, frame_rgb, frame_bgr):
        print("\n[Paused] Waiting for input...")
        
        # Show paused state
        frame_paused = frame_bgr.copy()
        cv2.putText(frame_paused, "PAUSED - Check Terminal", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("RealSense Feed", frame_paused)
        cv2.waitKey(1)
        
        try:
            user_input = input("Enter target object: ").strip()
        except EOFError:
            return

        if not user_input:
            print("Cancelled.")
            return
            
        timestamp = int(time.time())
        output_dir = Path(self.args.output).parent
        
        # Paths
        # 1. Original Image (Raw)
        path_original = output_dir / f"live_{timestamp}_{user_input}_original.jpg"
        # 2. Small Image (Resized)
        path_small = output_dir / f"live_{timestamp}_{user_input}_small.jpg"
        # 3. Visualization on Small Image
        path_small_result = output_dir / f"live_{timestamp}_{user_input}_small_result.jpg"
        # 4. Visualization on Original Image
        path_original_result = output_dir / f"live_{timestamp}_{user_input}_original_result.jpg"
        
        # Save Original
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(str(path_original))

        # Save Small (for Inference)
        pil_small = pil_img.copy()
        
        config_width = self.config.get("max_image_width", 640)
        should_resize = True
        
        if isinstance(config_width, str) and config_width.lower() == "default":
             should_resize = False
             target_width = None
        else:
            try:
                target_width = int(config_width)
            except ValueError:
                print(f"Warning: Invalid max_image_width '{config_width}', using default (no resize).")
                should_resize = False
                target_width = None
        
        if should_resize and target_width and pil_small.width > target_width:
            ratio = target_width / float(pil_small.width)
            new_height = int((float(pil_small.height) * float(ratio)))
            pil_small = pil_small.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        pil_small.save(str(path_small))
        
        print(f"Running detection on {path_small.name}...")
        
        # Define visualization tasks
        vis_tasks = [
            (str(path_small), str(path_small_result)),       # Box on small image
            (str(path_original), str(path_original_result))  # Box on original image
        ]
        
        self.run_detection(str(path_small), user_input, visualization_tasks=vis_tasks)
        
        # Show result in blocking window (prefer showing the high-res result)
        if path_original_result.exists():
            res_img = cv2.imread(str(path_original_result))
            if res_img is not None:
                cv2.imshow("RealSense Feed", res_img)
                print("Result displayed. Press any key to resume...")
                cv2.waitKey(0)
