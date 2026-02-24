"""L2 confirmation gate helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..day_trading_models import TradingSession
from ..strategies.base_strategy import Signal, SignalType


def passes_l2_confirmation(
    engine: Any,
    session: TradingSession,
    signal: Signal,
    flow_metrics: Optional[Dict[str, Any]] = None,
) -> tuple[bool, Dict[str, Any]]:
    """
    Optional order-flow confirmation gate.

    Uses only current/past bars (no look-ahead):
    - summed delta over lookback
    - mean imbalance over lookback
    - summed iceberg bias over lookback
    """
    flow_ctx = flow_metrics if isinstance(flow_metrics, dict) else {}
    strategy_key = engine._canonical_signal_strategy_key(signal)
    l2_book_pressure_z = engine._to_float(flow_ctx.get("l2_book_pressure_z"), 0.0)
    book_pressure_block_z_threshold = 2.5

    metrics: Dict[str, Any] = {
        "enabled": bool(session.l2_confirm_enabled),
        "strategy_key": strategy_key,
        "lookback_bars": max(1, int(session.l2_lookback_bars)),
        "min_delta": float(session.l2_min_delta),
        "min_imbalance": float(session.l2_min_imbalance),
        "min_iceberg_bias": float(session.l2_min_iceberg_bias),
        "min_participation_ratio": float(session.l2_min_participation_ratio),
        "min_directional_consistency": float(session.l2_min_directional_consistency),
        "min_signed_aggression": float(session.l2_min_signed_aggression),
        "l2_book_pressure_z": l2_book_pressure_z,
        "book_pressure_block_z_threshold": book_pressure_block_z_threshold,
        "passed": True,
    }

    if not session.l2_confirm_enabled:
        return True, metrics

    lookback = metrics["lookback_bars"]
    window = session.bars[-lookback:] if session.bars else []
    if not window:
        metrics.update({"passed": False, "reason": "l2_window_empty"})
        return False, metrics

    deltas = [engine._to_float(b.l2_delta, 0.0) for b in window]
    imbalances = [engine._to_float(b.l2_imbalance, 0.0) for b in window]
    iceberg_biases = [engine._to_float(b.l2_iceberg_bias, 0.0) for b in window]

    bars_with_l2 = sum(
        1 for b in window
        if (b.l2_delta is not None)
        or (b.l2_imbalance is not None)
        or (b.l2_iceberg_bias is not None)
    )
    has_any_l2 = bars_with_l2 > 0
    has_l2_coverage = bars_with_l2 >= max(3, len(window) // 2)
    metrics["has_l2_coverage"] = bool(has_l2_coverage)
    metrics["bars_with_l2"] = bars_with_l2
    if not has_any_l2:
        metrics.update({"passed": False, "reason": "l2_data_missing"})
        return False, metrics

    delta_sum = float(sum(deltas))
    imbalance_avg = float(sum(imbalances) / len(imbalances)) if imbalances else 0.0
    iceberg_bias_sum = float(sum(iceberg_biases))
    l2_volumes = [max(0.0, engine._to_float(b.l2_volume, 0.0)) for b in window]
    bar_volumes = [max(0.0, engine._to_float(b.volume, 0.0)) for b in window]

    direction = 1.0 if signal.signal_type == SignalType.BUY else -1.0
    directional_delta = delta_sum * direction
    directional_imbalance = imbalance_avg * direction
    directional_iceberg_bias = iceberg_bias_sum * direction

    participation_samples: List[float] = []
    signed_aggression_samples: List[float] = []
    directional_consistency_base = 0
    directional_consistency_hits = 0

    for b, l2_vol, bar_vol in zip(window, l2_volumes, bar_volumes):
        if l2_vol > 0 and bar_vol > 0:
            participation_samples.append(l2_vol / bar_vol)

        delta_val = engine._to_float(b.l2_delta, 0.0)
        if l2_vol > 0:
            signed_aggression_samples.append((delta_val / l2_vol) * direction)

        # Consistency check uses delta sign first, then imbalance sign fallback.
        if abs(delta_val) > 1e-9:
            directional_consistency_base += 1
            if (delta_val * direction) > 0:
                directional_consistency_hits += 1
            continue

        if b.l2_imbalance is not None:
            imb_val = engine._to_float(b.l2_imbalance, 0.0)
            if abs(imb_val) > 1e-9:
                directional_consistency_base += 1
                if (imb_val * direction) > 0:
                    directional_consistency_hits += 1

    participation_avg = (
        float(sum(participation_samples) / len(participation_samples))
        if participation_samples else 0.0
    )
    signed_aggression_avg = (
        float(sum(signed_aggression_samples) / len(signed_aggression_samples))
        if signed_aggression_samples else 0.0
    )
    directional_consistency = (
        float(directional_consistency_hits / directional_consistency_base)
        if directional_consistency_base > 0 else 0.0
    )

    min_delta_val = float(session.l2_min_delta)
    min_imbalance_val = float(session.l2_min_imbalance)
    min_iceberg_val = float(session.l2_min_iceberg_bias)
    min_participation_val = float(session.l2_min_participation_ratio)
    min_consistency_val = float(session.l2_min_directional_consistency)
    min_signed_agg_val = float(session.l2_min_signed_aggression)

    passes_delta = directional_delta >= min_delta_val
    passes_imbalance = directional_imbalance >= min_imbalance_val
    passes_iceberg = directional_iceberg_bias >= min_iceberg_val
    passes_participation = participation_avg >= min_participation_val
    passes_consistency = directional_consistency >= min_consistency_val
    passes_signed_aggression = signed_aggression_avg >= min_signed_agg_val

    l2_gate_mode = str(getattr(session, "l2_gate_mode", "weighted") or "weighted").lower()

    if l2_gate_mode == "weighted":
        def _l2_norm(value: float, threshold: float, scale: float = 2.0) -> float:
            if threshold <= 0:
                return 1.0 if value >= 0 else max(0.0, min(1.0, 0.5 + value))
            return max(0.0, min(1.0, value / (threshold * scale)))

        l2_gate_score = (
            0.25 * _l2_norm(directional_delta, max(min_delta_val, 1.0), 2.0)
            + 0.22 * _l2_norm(signed_aggression_avg, max(min_signed_agg_val, 0.01), 2.0)
            + 0.20 * _l2_norm(directional_consistency, max(min_consistency_val, 0.1), 2.0)
            + 0.20 * _l2_norm(directional_imbalance, max(min_imbalance_val, 0.01), 2.0)
            + 0.08 * _l2_norm(participation_avg, max(min_participation_val, 0.01), 2.0)
            + 0.05 * _l2_norm(directional_iceberg_bias, max(min_iceberg_val, 0.1), 2.0)
        )
        l2_gate_threshold = max(0.0, min(1.0, float(
            getattr(session, "l2_gate_threshold", 0.50) or 0.50
        )))
        passed = bool(l2_gate_score >= l2_gate_threshold)
    else:
        l2_gate_score = -1.0
        l2_gate_threshold = -1.0
        passed = bool(
            passes_delta
            and passes_imbalance
            and passes_iceberg
            and passes_participation
            and passes_consistency
            and passes_signed_aggression
        )

    metrics.update({
        "window_size": len(window),
        "l2_gate_mode": l2_gate_mode,
        "l2_gate_score": round(l2_gate_score, 4) if l2_gate_score >= 0 else None,
        "l2_gate_threshold": round(l2_gate_threshold, 4) if l2_gate_threshold >= 0 else None,
        "delta_sum": delta_sum,
        "imbalance_avg": imbalance_avg,
        "iceberg_bias_sum": iceberg_bias_sum,
        "participation_avg": participation_avg,
        "directional_consistency": directional_consistency,
        "signed_aggression_avg": signed_aggression_avg,
        "directional_delta": directional_delta,
        "directional_imbalance": directional_imbalance,
        "directional_iceberg_bias": directional_iceberg_bias,
        "passes_delta": passes_delta,
        "passes_imbalance": passes_imbalance,
        "passes_iceberg": passes_iceberg,
        "passes_participation": passes_participation,
        "passes_consistency": passes_consistency,
        "passes_signed_aggression": passes_signed_aggression,
    })

    soft_gate_passed = bool(passed)
    is_long_signal = signal.signal_type == SignalType.BUY
    is_short_signal = signal.signal_type == SignalType.SELL

    book_pressure_block_long = (
        is_long_signal and l2_book_pressure_z < -book_pressure_block_z_threshold
    )
    book_pressure_block_short = (
        is_short_signal and l2_book_pressure_z > book_pressure_block_z_threshold
    )

    hard_block_reason: Optional[str] = None
    if book_pressure_block_long:
        hard_block_reason = "book_pressure_block_long"
    elif book_pressure_block_short:
        hard_block_reason = "book_pressure_block_short"
    # L2 acts as an emergency brake only; soft score diagnostics stay informational.
    passed = bool(hard_block_reason is None)
    metrics.update(
        {
            "l2_effective_mode": "hard_block_only",
            "l2_soft_gate_passed": soft_gate_passed,
            "book_pressure_block_long": bool(book_pressure_block_long),
            "book_pressure_block_short": bool(book_pressure_block_short),
            "passes_book_pressure_block": bool(
                not book_pressure_block_long and not book_pressure_block_short
            ),
            "hard_block": bool(hard_block_reason),
            "passed": passed,
        }
    )

    if hard_block_reason:
        metrics["reason"] = hard_block_reason
    return passed, metrics
