from typing import Dict, Any, Union, List
from src.core.llm_client import LLMClient
from src.parsers.bbox_parser import BaseParser
import time

class ObjectDetector:
    def __init__(self, llm_client: LLMClient, parser: BaseParser):
        self.llm = llm_client
        self.parser = parser

    def run(self, image: Union[str, bytes], prompt: Union[str, Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Run detection on an image (path or bytes).
        """
        start_time = time.time()
        
        # 1. Generate
        # Ollama expects a list of images
        raw_response = self.llm.generate(prompt, images=[image], **kwargs)
        
        # 2. Parse
        coordinates = self.parser.parse(raw_response)
        
        latency = time.time() - start_time
        
        return {
            "success": True,
            "prompt": prompt,
            "raw_response": raw_response,
            "coordinates": coordinates,
            "latency_seconds": latency
        }

