import argparse
import os
import sys
from pathlib import Path

# Fix: Unset proxy environment variables that cause issues with httpx/ollama
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
     if k in os.environ:
         del os.environ[k]

from src.core.config import load_config
from src.apps.static import StaticDetectionApp

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Ollama VLM - Static Image Mode")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target", type=str, help="Target object description")
    parser.add_argument("--output", type=str, help="Path to save result image (Optional)")

    args = parser.parse_args()
    
    image_path = args.image
    target_object = args.target
    output_directory = config.get("output_dir", "output")
    
    print("\n=== Static Detection Configuration ===")
    print(f"Image: {image_path}")
    print(f"Target: {target_object or '[Interactive Input]'}")
    print(f"Model: {config.get('default_model', 'qwen3-vl')}")
    print("======================================\n")

    # Determine Output Path
    final_output_path = args.output
    if not final_output_path and output_directory:
        out_dir_path = Path(output_directory)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        src_name = Path(image_path).stem
        final_output_path = str(out_dir_path / f"{src_name}_result.jpg")

    app = StaticDetectionApp(
        model_name=config.get("default_model", "qwen3-vl"),
        template_name=config.get("template", "standard_detection.v2"),
        prompts_dir=config.get("prompts_dir", "prompts"),
        visualization_config=config.get("visualization", {})
    )
    
    app.run(image_path=image_path, target_object=target_object, output_path=final_output_path)



if __name__ == "__main__":
    main()
