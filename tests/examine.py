import json
from typing import Union, Any, Iterable
from pathlib import Path

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

jsonl_path1 = ""
jsonl_path2 = ""

