"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FaultConfig:
    permission_path: Optional[str] = None
    missing_path: Optional[str] = None
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    tool_timeout_s: Optional[float] = None


@dataclass
class ExperimentConfig:
    orchestrators: List[str] = field(default_factory=lambda: ["langgraph"])
    runtimes: List[str] = field(default_factory=lambda: ["ollama"])
    tasks: List[str] = field(default_factory=lambda: ["all"])
    seeds: List[int] = field(default_factory=lambda: [1])
    max_steps: int = 8
    max_llm_retries: int = 2
    max_tool_retries: int = 1
    timeout_s: float = 20.0
    tool_timeout_s: float = 10.0
    results_dir: str = "evaluation/results"
    sandbox_dir: str = "evaluation/sandboxes"
    transport: str = "stdio"
    gateway_cmd: Optional[List[str]] = None
    http_url: Optional[str] = None
    runtime_models: Dict[str, str] = field(default_factory=dict)
    faults: FaultConfig = field(default_factory=FaultConfig)


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [os.path.expandvars(item) if isinstance(item, str) else item for item in value]
    return value


def load_config(path: Optional[Path]) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for key, value in data.items():
        if key == "faults" and isinstance(value, dict):
            for f_key, f_value in value.items():
                if hasattr(config.faults, f_key):
                    setattr(config.faults, f_key, _expand_value(f_value))
            continue
        if hasattr(config, key):
            setattr(config, key, _expand_value(value))
    return config
