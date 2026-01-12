import ollama
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

class OllamaDetector:
    def __init__(self, model_name: str = "qwen3vl-32B"):
        """
        初始化检测器
        
        Args:
            model_name: Ollama中运行的模型名称
        """
        self.model = model_name

    def detect(self, image_path: str, target_object: str, prompt_template: str = "Detect {}") -> Dict[str, Any]:
        """
        检测图片中的目标物体及其坐标
        
        Args:
            image_path: 图片文件的路径
            target_object: 需要检测的目标物体描述
            prompt_template: 用于构建Prompt的模板，必须包含 {} 供 target_object 填充
            
        Returns:
            Dict containing the raw response and parsed coordinates (list of boxes)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Qwen-VL 系列模型通常对 "Detect {object}" 或 "Find {object}" 这种指令响应较好
        # 我们可以尝试引导它输出具体的坐标格式
        try:
            prompt = prompt_template.format(target_object)
        except:
            prompt = f"Detect {target_object}"

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ]
            )
            
            content = response.get('message', {}).get('content', '')
            
            # 解析结果
            coordinates = self._parse_coordinates(content)
            
            return {
                "success": True,
                "target": target_object,
                "raw_response": content,
                "coordinates": coordinates # List[List[int]]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _parse_coordinates(self, content: str) -> List[List[int]]:
        """
        尝试从响应中解析坐标
        
        Returns:
            List of [y1, x1, y2, x2] (normalized 0-1000)
        """
        boxes = []

        # 1. 优先尝试匹配 JSON 格式 (e.g. {"bbox_2d": [x, y, x, y]})
        try:
            json_bbox_pattern = r'"bbox_2d":\s*\[([\d,\s]+)\]'
            json_matches = re.findall(json_bbox_pattern, content)
            
            if json_matches:
                for match in json_matches:
                    nums = [int(n) for n in re.findall(r'\d+', match)]
                    if len(nums) == 4:
                        # Assume JSON bbox_2d is [x1, y1, x2, y2]
                        # Convert to [y1, x1, y2, x2] to match Qwen standard
                        x1, y1, x2, y2 = nums
                        boxes.append([y1, x1, y2, x2])
                if boxes:
                    return boxes
        except Exception:
            pass

        # 2. 尝试匹配 <box>...</box> 模式
        box_pattern = r'<box>(.*?)</box>'
        box_matches = re.findall(box_pattern, content)
        
        if box_matches:
            for match in box_matches:
                nums = re.findall(r'\d+', match)
                if len(nums) == 4:
                    # Qwen-VL default <box> format: y1, x1, y2, x2
                    boxes.append([int(n) for n in nums])
            if boxes:
                return boxes
        
        # 备用：搜索文本中连续出现的4个数字 (仅当没找到其他格式时)
        # 这种比较宽泛，可能会误判，暂时略过防止噪音
        
        return []
