"""Dependency-free parsing of CLI search arguments."""

from __future__ import annotations

import json


def parse_value(text: str):
    if ":" in text:
        lower, upper = text.split(":", 1)
        try:
            return (
                None if lower == "" else float(lower),
                None if upper == "" else float(upper),
            )
        except ValueError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_filters(items):
    filters = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter {item!r} must have the form NAME=VALUE.")
        name, value = item.split("=", 1)
        filters[name] = parse_value(value)
    return filters

