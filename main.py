import argparse
from pathlib import Path

from src.core.config import load_config
from src.application import LiveDetectionApp

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Ollama VLM Prompt Engineering Framework")
    # Kept previous arguments for compatibility/configuration, even if live mode is default
    parser.add_argument("--image", type=str, default="example_data/1_Color.png", help="Path to input image (Unused in Live Mode)")
    parser.add_argument("--target", type=str, help="Description of the target object")
    parser.add_argument("--template", type=str, default="standard_detection.v2", help="Prompt template key")
    parser.add_argument("--model", type=str, default=config.get("default_model", "qwen3-vl"), help="Ollama model name")
    parser.add_argument("--output", type=str, default=f"{config.get('output_dir', 'output')}/output_test.jpg", help="Path to save result image")
    parser.add_argument("--experiment_name", type=str, default="default", help="Tag for specific experiment run")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run App
    app = LiveDetectionApp(config, args)
    app.run()

if __name__ == "__main__":
    main()
