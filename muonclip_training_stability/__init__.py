"""Deterministic reference semantics for Chapter 13."""

from .reference import (
    MLAHead,
    clip_factor,
    distributed_consensus,
    large_logit_ratio,
    lag_counterexample,
    mha_allocation_factors,
    mla_clip,
    mla_score,
    running_trace,
    scaled_positive_max,
    soft_cap,
    trigger_statistics,
)

__all__ = [
    "MLAHead",
    "clip_factor",
    "distributed_consensus",
    "large_logit_ratio",
    "lag_counterexample",
    "mha_allocation_factors",
    "mla_clip",
    "mla_score",
    "running_trace",
    "scaled_positive_max",
    "soft_cap",
    "trigger_statistics",
]
