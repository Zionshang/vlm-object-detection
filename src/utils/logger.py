import json
import datetime
from pathlib import Path
from typing import Any, Dict

class ExperimentLogger:
    def __init__(self, log_file: str = "experiment_logs.jsonl"):
        self.log_file = Path(log_file)

    def log(self, run_info: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            **run_info
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
