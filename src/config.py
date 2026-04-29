from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PipelinePaths:
    train_feature: str = "data/train/mengxi_boundary_anon_filtered.csv"
    train_label: str = "data/train/mengxi_node_price_selected.csv"
    test_feature: str = "data/test/test_in_feature_ori.csv"
    model_output: str = "outputs/lgb_model.txt"
    metadata_output: str = "outputs/lgb_model_metadata.json"
    predictions_output: str = "outputs/test_predictions.csv"
    submission_output: str = "output.csv"


def load_simple_yaml(path: str) -> Dict[str, Any]:
    """Load a tiny key/value YAML file without requiring PyYAML.

    This is intentionally small: it supports nested dictionaries by indentation
    and scalar strings/numbers, which is enough for configs/default.yaml.
    """
    result: Dict[str, Any] = {}
    stack = [(0, result)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        value = value.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent + 2, child))
        else:
            parent[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip("'\"")
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

