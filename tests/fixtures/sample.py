"""
Fixture: sample Python module for lexical parser tests.
Intentionally includes classes, functions, imports, and TODO comments.
"""

import os
import json
from pathlib import Path

# TODO: Add caching layer here
# FIXME: This class is not thread-safe

class DataProcessor:
    """Processes raw data payloads."""

    def __init__(self, config: dict):
        self.config = config

    def load(self, filepath: str) -> dict:
        """Load JSON from disk."""
        with open(filepath) as f:
            return json.load(f)

    def transform(self, data: dict) -> dict:
        # BUG: empty dicts are not handled
        return {k: v.strip() for k, v in data.items()}


class DataWriter:
    """Writes processed data back to disk."""

    def write(self, data: dict, output_path: str) -> None:
        Path(output_path).write_text(json.dumps(data, indent=2))


def validate_schema(data: dict, schema: dict) -> bool:
    """Check that all required keys are present."""
    return all(k in data for k in schema.get("required", []))


def run_pipeline(config_path: str) -> None:
    """Entry point for the data pipeline."""
    # TODO: Add retry logic
    processor = DataProcessor(config={})
    data = processor.load(config_path)
    validate_schema(data, {})
