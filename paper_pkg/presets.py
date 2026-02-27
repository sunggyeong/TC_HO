#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper_pkg.presets

Shared presets for training penalties + realtime guardrails + baseline guardrails.
"""
from __future__ import annotations

PRESETS = {
    "stable": {
        "w_switch": 0.25, "w_pingpong": 0.80, "w_jitter": 0.45,
        "rt_dwell": 4, "rt_hysteresis": 0.12, "rt_pingpong_window": 6, "rt_pingpong_extra": 0.18,
        "baseline_dwell": 4, "baseline_hysteresis": 0.12, "baseline_pingpong_window": 6, "baseline_pingpong_extra": 0.18,
        "alpha_latency": 0.05, "lookahead_H": 30, "gamma": 0.97, "shoot_K": 256,
    },
    "balanced": {
        "w_switch": 0.18, "w_pingpong": 0.60, "w_jitter": 0.35,
        "rt_dwell": 3, "rt_hysteresis": 0.10, "rt_pingpong_window": 5, "rt_pingpong_extra": 0.15,
        "baseline_dwell": 3, "baseline_hysteresis": 0.10, "baseline_pingpong_window": 5, "baseline_pingpong_extra": 0.15,
        "alpha_latency": 0.05, "lookahead_H": 30, "gamma": 0.97, "shoot_K": 256,
    },
    "aggressive": {
        "w_switch": 0.12, "w_pingpong": 0.45, "w_jitter": 0.25,
        "rt_dwell": 2, "rt_hysteresis": 0.07, "rt_pingpong_window": 4, "rt_pingpong_extra": 0.10,
        "baseline_dwell": 2, "baseline_hysteresis": 0.07, "baseline_pingpong_window": 4, "baseline_pingpong_extra": 0.10,
        "alpha_latency": 0.05, "lookahead_H": 30, "gamma": 0.97, "shoot_K": 256,
    },
}
