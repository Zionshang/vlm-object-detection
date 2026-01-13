import argparse
import sys
import yaml
from pathlib import Path

# Framework imports
from src.core.llm_client import OllamaClient
from src.core.prompt_manager import PromptManager
from src.parsers.bbox_parser import BoundingBoxParser
from src.pipeline import DetectionPipeline
from src.utils.image_utils import draw_bounding_boxes
from src.utils.logger import ExperimentLogger

def load_config(config_path="config/settings.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Ollama VLM Prompt Engineering Framework")
    parser.add_argument("--image", type=str, default="example_data/1_Color.png", help="Path to input image")
    parser.add_argument("--target", type=str, required=True, help="Description of the target object")
    parser.add_argument("--template", type=str, default="standard_detection.v2", help="Prompt template key (e.g. standard_detection.v2)")
    parser.add_argument("--model", type=str, default=config.get("default_model", "qwen3-vl"), help="Ollama model name")
    parser.add_argument("--output", type=str, default=f"{config.get('output_dir', 'output')}/output_test.jpg", help="Path to save result image")
    parser.add_argument("--experiment_name", type=str, default="default", help="Tag for specific experiment run")

    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Initializing Experiment: {args.experiment_name}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    
    # 2. Components
    llm_client = OllamaClient(model_name=args.model)
    prompt_manager = PromptManager(prompt_dir=config.get("prompts_dir", "prompts"))
    bbox_parser = BoundingBoxParser()
    pipeline = DetectionPipeline(llm_client, bbox_parser)
    logger = ExperimentLogger(log_file=config.get("log_file", "logs/experiments.jsonl"))

    # 3. Prepare Prompt
    try:
        prompt = prompt_manager.format_prompt(args.template, target=args.target)
        print(f"Constructed Prompt: {prompt}")
    except ValueError as e:
        print(f"Error constructing prompt: {e}")
        sys.exit(1)

    # 4. Execute Pipeline
    print("Running pipeline...")
    result = pipeline.run(str(image_path), prompt)

    # 5. Log & Visualize
    result['experiment_name'] = args.experiment_name
    result['model'] = args.model
    result['template_key'] = args.template
    result['target'] = args.target
    result['image_path'] = str(image_path)
    
    logger.log(result)
    
    print("-" * 30)
    print("Raw Response:")
    print(result['raw_response'])
    print("-" * 30)
    if config.get("show_latency", False):
        print(f"Inference Latency: {result.get('latency_seconds', 0):.4f}s")
        print("-" * 30)

    
    coords = result.get('coordinates', [])
    if coords:
        print(f"Found {len(coords)} objects.")
        labels = [f"{args.target} {i+1}" for i in range(len(coords))]
        draw_bounding_boxes(str(image_path), coords, args.output, labels)
    else:
        print("No objects detected or parsing failed.")

if __name__ == "__main__":
    main()
