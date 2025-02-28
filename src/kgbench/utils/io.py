import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def save_jsonl(data: List, jsonl_path: str) -> None:
    """
    Utility method to serialize a list of json objects to a .jsonl file
    :param data: list of data items
    :param jsonl_path: path to the output .jsonl file
    :return: None
    """
    # ensure directory exists
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding='utf-8') as out_file:
        for item in data:
            out_file.write(f"{json.dumps(item)}\n")


def append_jsonl(data: Dict, jsonl_path: str) -> None:
    """
    Utility method to append a new line to a .jsonl file
    :param data: data to be serialized into the file
    :param jsonl_path: path of the file to be appended
    :return: None
    """
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a+", encoding='utf-8') as out_file:
        out_file.write(f"{json.dumps(data)}\n")


def read_json(src_file: str) -> Optional[Union[Dict,List]]:
    """Load either JSON or JSONL file"""
    try:
        with open(src_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                f.seek(0)
                data = []
                for line in f:
                    data.append(json.loads(line.strip()))
                return data
    except Exception as e:
        print(f"Error loading file {src_file}: {str(e)}")
        return None
