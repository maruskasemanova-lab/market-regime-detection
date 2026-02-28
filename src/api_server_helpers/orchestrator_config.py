"""Helpers for serializing and mutating orchestrator runtime config."""

from __future__ import annotations

from typing import Any, Dict, Mapping


_FEATURE_FLAG_FIELDS = (
    "use_evidence_engine",
    "use_adaptive_regime",
    "use_calibration",
    "use_quality_sizing",
    "use_cross_asset",
    "use_edge_monitor",
)

_NUMERIC_FIELDS = (
    "min_confirming_sources",
    "base_threshold",
    "base_risk_pct",
    "min_margin_over_threshold",
    "single_source_min_margin",
    "strategy_weight",
    "strategy_only_threshold",
)

_RUNTIME_PROPAGATION_FIELDS = (
    "base_threshold",
    "min_confirming_sources",
    "min_margin_over_threshold",
    "single_source_min_margin",
)


def serialize_orchestrator_config(orch: Any) -> Dict[str, Any]:
    cfg = orch.config
    combiner = orch.combiner if hasattr(orch, "combiner") else None

    return {
        "use_evidence_engine": cfg.use_evidence_engine,
        "use_adaptive_regime": cfg.use_adaptive_regime,
        "use_calibration": cfg.use_calibration,
        "use_quality_sizing": cfg.use_quality_sizing,
        "use_cross_asset": cfg.use_cross_asset,
        "use_edge_monitor": cfg.use_edge_monitor,
        "min_confirming_sources": cfg.min_confirming_sources,
        "base_threshold": cfg.base_threshold,
        "base_risk_pct": cfg.base_risk_pct,
        "combiner_base_threshold": combiner._base_threshold if combiner else None,
        "combiner_min_confirming": combiner._min_confirming if combiner else None,
        "combiner_min_margin": combiner._min_margin if combiner else None,
        "combiner_single_source_margin": combiner._single_source_margin if combiner else None,
        "strategy_weight": cfg.strategy_weight,
        "strategy_only_threshold": cfg.strategy_only_threshold,
    }


def apply_orchestrator_config_updates(
    *,
    orch: Any,
    body: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = orch.config
    updated: Dict[str, Any] = {}

    for flag in _FEATURE_FLAG_FIELDS:
        if flag in body:
            value = bool(body[flag])
            setattr(cfg, flag, value)
            updated[flag] = value

    for param in _NUMERIC_FIELDS:
        if param in body:
            value = int(body[param]) if param == "min_confirming_sources" else float(body[param])
            setattr(cfg, param, value)
            updated[param] = value

    # Keep downstream thresholds in sync immediately (combiner/evidence engine
    # hold their own runtime copies and do not re-read config every decision).
    if any(field in body for field in _RUNTIME_PROPAGATION_FIELDS):
        if hasattr(orch, "combiner") and orch.combiner is not None:
            if "base_threshold" in body:
                orch.combiner._base_threshold = float(body["base_threshold"])
            if "min_confirming_sources" in body:
                orch.combiner._min_confirming = int(body["min_confirming_sources"])
            if "min_margin_over_threshold" in body:
                orch.combiner._min_margin = float(body["min_margin_over_threshold"])
            if "single_source_min_margin" in body:
                orch.combiner._single_source_margin = float(body["single_source_min_margin"])
        if hasattr(orch, "evidence_engine") and orch.evidence_engine is not None:
            if "base_threshold" in body:
                orch.evidence_engine._base_threshold = float(body["base_threshold"])
            if "min_confirming_sources" in body:
                orch.evidence_engine._min_confirming = int(body["min_confirming_sources"])

    if "strategy_only_threshold" in body:
        if hasattr(orch, "evidence_engine") and orch.evidence_engine is not None:
            orch.evidence_engine._strategy_only_threshold = float(body["strategy_only_threshold"])

    return updated


__all__ = [
    "apply_orchestrator_config_updates",
    "serialize_orchestrator_config",
]
