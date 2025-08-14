import os
import csv
from typing import Dict

class RunLogger:
    def __init__(self, csv_path: str, header: list):
        self.csv_path = csv_path
        self.header = header
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def append(self, row: Dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            w.writerow(row)
