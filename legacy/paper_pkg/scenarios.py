#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pkg.scenarios

Only two scenario presets used in the paper:
  - normal       : cfg.stress_profile="easy"
  - paper_stress : cfg.stress_profile="paper_stress"
"""
from __future__ import annotations
from typing import Literal
from .env_core import ExperimentConfig

ScenarioName = Literal["normal", "paper_stress"]

def apply_scenario(cfg: ExperimentConfig, scenario: str) -> ExperimentConfig:
    s = (scenario or "normal").strip().lower()
    if s in ("normal", "base", "easy", "typical"):
        cfg.stress_profile = "easy"
        cfg.stress_decision_budget_ms_override = None
        return cfg
    if s in ("paper_stress", "stress_paper", "stress"):
        cfg.stress_profile = "paper_stress"
        return cfg
    raise ValueError(f"Unknown scenario='{scenario}'. Use 'normal' or 'paper_stress'.")
