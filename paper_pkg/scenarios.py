#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.scenarios

Only two scenarios used in the paper:
- normal: easy / typical
- paper_stress: stress setting used for main paper figures
"""
from __future__ import annotations
from typing import Literal
from .env_core import ExperimentConfig

ScenarioName = Literal["normal", "paper_stress"]

def apply_scenario(cfg: ExperimentConfig, scenario: str) -> ExperimentConfig:
    s = (scenario or "normal").strip().lower()
    if s in ("normal", "easy", "base", "typical"):
        cfg.stress_profile = "easy"
        cfg.stress_decision_budget_ms_override = None
        # DAG: allow skip edges helps with sporadic outages too
        cfg.dag_allow_skip_edge = True
        return cfg
    if s in ("paper_stress", "stress", "stress_paper"):
        cfg.stress_profile = "paper_stress"
        # Stress has many outages; allow skip edges so DAG remains a meaningful baseline
        cfg.dag_allow_skip_edge = True
        return cfg
    raise ValueError("scenario must be one of: normal, paper_stress")
