"""CSV/JSON parsing helpers shared by experiment workflows."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np
import pandas as pd


def parse_feature_names(value: Any) -> list[str]:
    """Parse feature names from JSON, Python literals, comma text, or arrays."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist() if str(item)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        return [str(value)] if str(value) else []

    clean = value.strip()
    if not clean:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(clean)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list | tuple):
            return [str(item) for item in parsed if str(item)]
    if "," in clean:
        return [part.strip() for part in clean.split(",") if part.strip()]
    return [clean]


def parse_feature_list(value: Any) -> list[str]:
    """Parse CSV-safe feature-list values, rejecting plain scalar text."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if not isinstance(value, str) or not value.strip():
        return []
    clean = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(clean)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list | tuple):
            return [str(item) for item in parsed]
    return []


def parse_json_list(value: Any) -> list[str]:
    """Parse JSON/list values from recipe tables."""
    return parse_feature_names(value)


def parse_json_dict(value: Any) -> dict[str, Any]:
    """Parse JSON or Python-literal mapping values from recipe tables."""
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    clean = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(clean)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}
