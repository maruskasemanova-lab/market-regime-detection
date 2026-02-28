"""Order-flow metric helpers used by regime selection."""

from __future__ import annotations

import math
from typing import Dict, List

from ..day_trading_models import BarData

def _flow_score_from_components(
    self,
    *,
    cumulative_delta: float,
    deltas: List[float],
    directional_consistency: float,
    avg_imbalance: float,
    sweep_intensity: float,
    participation_ratio: float,
    large_trader_activity: float,
    vwap_execution_flow: float,
    book_pressure_avg: float,
) -> float:
    """
    Composite flow quality score in 0-100 range.

    Uses only scale-free ratios so values are comparable across tickers and
    volume regimes.
    """
    delta_efficiency = self._safe_div(
        abs(cumulative_delta),
        sum(abs(d) for d in deltas),
        0.0,
    )
    return 100.0 * (
        0.22 * max(0.0, min(1.0, delta_efficiency))
        + 0.19 * max(0.0, min(1.0, directional_consistency))
        + 0.15 * max(0.0, min(1.0, abs(avg_imbalance)))
        + 0.11 * max(0.0, min(1.0, sweep_intensity))
        + 0.10 * max(0.0, min(1.0, participation_ratio))
        + 0.08 * max(0.0, min(1.0, large_trader_activity))
        + 0.07 * max(0.0, min(1.0, vwap_execution_flow))
        + 0.08 * max(0.0, min(1.0, abs(book_pressure_avg)))
    )


def _calculate_window_flow_components(self, window: List[BarData]) -> Dict[str, float]:
    """Compute flow metrics for one contiguous bar window."""
    bars_with_l2 = 0
    deltas: List[float] = []
    imbalances: List[float] = []
    iceberg_biases: List[float] = []
    l2_volumes: List[float] = []
    book_pressures: List[float] = []
    book_pressure_changes: List[float] = []
    bid_depth_totals: List[float] = []
    ask_depth_totals: List[float] = []
    bar_volumes: List[float] = []
    close_to_close_pct: List[float] = []
    vwap_cluster_l2_volume = 0.0
    for i, b in enumerate(window):
        has_l2 = (
            b.l2_delta is not None
            or b.l2_imbalance is not None
            or b.l2_volume is not None
            or b.l2_book_pressure is not None
            or b.l2_bid_depth_total is not None
            or b.l2_ask_depth_total is not None
            or b.l2_iceberg_bias is not None
            or getattr(b, "tcbbo_has_data", False)
            or getattr(b, "tcbbo_net_premium", None) is not None
        )
        if has_l2:
            bars_with_l2 += 1

        deltas.append(self._to_float(b.l2_delta, 0.0))
        imbalances.append(self._to_float(b.l2_imbalance, 0.0))
        iceberg_biases.append(self._to_float(b.l2_iceberg_bias, 0.0))
        l2_volumes.append(max(0.0, self._to_float(b.l2_volume, 0.0)))
        book_pressures.append(self._to_float(b.l2_book_pressure, 0.0))
        book_pressure_changes.append(self._to_float(b.l2_book_pressure_change, 0.0))
        bid_depth_totals.append(max(0.0, self._to_float(b.l2_bid_depth_total, 0.0)))
        ask_depth_totals.append(max(0.0, self._to_float(b.l2_ask_depth_total, 0.0)))
        bar_volumes.append(max(0.0, self._to_float(b.volume, 0.0)))
        if b.vwap is not None and b.close > 0:
            vwap_distance_pct = abs((b.close - b.vwap) / b.close) * 100.0
            if vwap_distance_pct <= 0.05:
                vwap_cluster_l2_volume += l2_volumes[-1]

        if i > 0 and window[i - 1].close > 0:
            close_to_close_pct.append((b.close - window[i - 1].close) / window[i - 1].close * 100.0)

    has_l2_coverage = bars_with_l2 >= max(3, len(window) // 2)
    cumulative_delta = float(sum(deltas))
    avg_imbalance = float(sum(imbalances) / len(imbalances)) if imbalances else 0.0
    iceberg_bias_sum = float(sum(iceberg_biases))
    total_l2_volume = float(sum(l2_volumes))
    total_bar_volume = float(sum(bar_volumes))
    participation_ratio = self._safe_div(total_l2_volume, total_bar_volume, 0.0)
    signed_aggression = self._safe_div(cumulative_delta, total_l2_volume, 0.0)
    avg_bid_depth_total = float(sum(bid_depth_totals) / len(bid_depth_totals)) if bid_depth_totals else 0.0
    avg_ask_depth_total = float(sum(ask_depth_totals) / len(ask_depth_totals)) if ask_depth_totals else 0.0
    book_pressure_avg = float(sum(book_pressures) / len(book_pressures)) if book_pressures else 0.0
    book_pressure_change_avg = (
        float(sum(book_pressure_changes) / len(book_pressure_changes))
        if book_pressure_changes else 0.0
    )
    if len(book_pressures) >= 2:
        book_pressure_trend = float(book_pressures[-1] - book_pressures[0])
    else:
        book_pressure_trend = float(book_pressure_change_avg)

    first_close = window[0].close
    last_close = window[-1].close
    price_change_pct = self._safe_div((last_close - first_close) * 100.0, first_close, 0.0)
    total_price_move = sum(
        abs(window[i].close - window[i - 1].close)
        for i in range(1, len(window))
    )
    price_trend_efficiency = self._safe_div(abs(last_close - first_close), total_price_move, 0.0)

    directional_base = 0
    directional_hits = 0
    low_progress_l2_volume = 0.0
    for i in range(1, len(window)):
        delta_val = deltas[i]
        prev_close = window[i - 1].close
        if prev_close <= 0:
            continue
        price_change = (window[i].close - prev_close) / prev_close * 100.0
        if abs(delta_val) > 1e-9 and abs(price_change) > 1e-6:
            directional_base += 1
            if (delta_val * price_change) > 0:
                directional_hits += 1
        if abs(price_change) <= 0.02:
            low_progress_l2_volume += l2_volumes[i]

    directional_consistency = self._safe_div(float(directional_hits), float(directional_base), 0.0)
    absorption_rate = self._safe_div(low_progress_l2_volume, total_l2_volume, 0.0)

    abs_deltas = [abs(v) for v in deltas]
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    delta_variance = (
        sum((ad - mean_abs_delta) ** 2 for ad in abs_deltas) / len(abs_deltas)
        if abs_deltas else 0.0
    )
    delta_std = math.sqrt(delta_variance) if delta_variance > 0 else 0.0
    avg_l2_volume = sum(l2_volumes) / len(l2_volumes) if l2_volumes else 0.0
    l2_volume_variance = (
        sum((lv - avg_l2_volume) ** 2 for lv in l2_volumes) / len(l2_volumes)
        if l2_volumes else 0.0
    )
    l2_volume_std = math.sqrt(l2_volume_variance) if l2_volume_variance > 0 else 0.0
    large_trade_threshold = avg_l2_volume + l2_volume_std
    sweep_hits = 0
    large_trade_hits = 0
    for d, lv in zip(abs_deltas, l2_volumes):
        if d >= (mean_abs_delta + delta_std) and lv >= (avg_l2_volume * 1.2):
            sweep_hits += 1
        if lv >= large_trade_threshold:
            large_trade_hits += 1
    sweep_intensity = self._safe_div(float(sweep_hits), float(len(window)), 0.0)
    large_trader_activity = self._safe_div(float(large_trade_hits), float(len(window)), 0.0)
    vwap_execution_flow = self._safe_div(vwap_cluster_l2_volume, total_l2_volume, 0.0)

    delta_mean = sum(deltas) / len(deltas) if deltas else 0.0
    delta_var = (
        sum((d - delta_mean) ** 2 for d in deltas) / len(deltas)
        if deltas else 0.0
    )
    delta_sigma = math.sqrt(delta_var) if delta_var > 0 else 0.0
    last_delta = deltas[-1] if deltas else 0.0
    delta_zscore = self._safe_div(last_delta - delta_mean, delta_sigma, 0.0)

    realized_volatility_pct = 0.0
    if close_to_close_pct:
        mean_ret = sum(close_to_close_pct) / len(close_to_close_pct)
        var_ret = sum((r - mean_ret) ** 2 for r in close_to_close_pct) / len(close_to_close_pct)
        realized_volatility_pct = math.sqrt(var_ret)
    vol_floor = max(realized_volatility_pct, 0.05)
    normalized_price = price_change_pct / vol_floor
    delta_price_divergence = signed_aggression - normalized_price

    flow_score = _flow_score_from_components(
        self,
        cumulative_delta=cumulative_delta,
        deltas=deltas,
        directional_consistency=directional_consistency,
        avg_imbalance=avg_imbalance,
        sweep_intensity=sweep_intensity,
        participation_ratio=participation_ratio,
        large_trader_activity=large_trader_activity,
        vwap_execution_flow=vwap_execution_flow,
        book_pressure_avg=book_pressure_avg,
    )

    latest_bar = window[-1]
    latest_range = max(1e-9, latest_bar.high - latest_bar.low)
    latest_bar_body_ratio = max(
        0.0,
        min(1.0, abs(latest_bar.close - latest_bar.open) / latest_range),
    )
    latest_bar_close_location = max(
        0.0,
        min(1.0, (latest_bar.close - latest_bar.low) / latest_range),
    )

    return {
        "has_l2_coverage": bool(has_l2_coverage),
        "bars_with_l2": float(bars_with_l2),
        "lookback_bars": float(len(window)),
        "cumulative_delta": cumulative_delta,
        "delta_zscore": delta_zscore,
        "imbalance_avg": avg_imbalance,
        "iceberg_bias": iceberg_bias_sum,
        "participation_ratio": participation_ratio,
        "directional_consistency": directional_consistency,
        "signed_aggression": signed_aggression,
        "absorption_rate": absorption_rate,
        "book_pressure_avg": book_pressure_avg,
        "book_pressure_trend": book_pressure_trend,
        "book_pressure_change_avg": book_pressure_change_avg,
        "bid_depth_total_avg": avg_bid_depth_total,
        "ask_depth_total_avg": avg_ask_depth_total,
        "sweep_intensity": sweep_intensity,
        "large_trader_activity": large_trader_activity,
        "vwap_execution_flow": vwap_execution_flow,
        "price_change_pct": price_change_pct,
        "price_trend_efficiency": price_trend_efficiency,
        "latest_bar_body_ratio": latest_bar_body_ratio,
        "latest_bar_close_location": latest_bar_close_location,
        "realized_volatility_pct": realized_volatility_pct,
        "delta_price_divergence": delta_price_divergence,
        "flow_score": flow_score,
    }


def regime_calculate_order_flow_metrics(
    self,
    bars: List[BarData],
    lookback: int = 20,
) -> Dict[str, float]:
    """Calculate no-lookahead microstructure metrics from existing L2-enriched bars."""
    window_size = max(5, int(lookback))
    window = bars[-window_size:] if bars else []

    if not window:
        return {
            "has_l2_coverage": False,
            "bars_with_l2": 0.0,
            "lookback_bars": float(window_size),
        }
    current_metrics = _calculate_window_flow_components(self, window)

    # ── Multi-bar flow trending (P1-A) ──────────────────────────────
    # Compare current flow metrics against a 3-bar-ago snapshot to detect
    # acceleration/deterioration trends. Uses bars available before the
    # current lookback window to avoid overlap.
    prev_window = bars[-(window_size * 2): -window_size] if len(bars) > window_size else []
    prev_delta = float(sum(self._to_float(b.l2_delta, 0.0) for b in prev_window)) if prev_window else 0.0
    delta_acceleration = float(current_metrics.get("cumulative_delta", 0.0)) - prev_delta

    trend_bars = 3
    flow_score_trend = 0.0
    signed_aggression_trend = 0.0
    book_pressure_trend_3bar = 0.0

    if len(bars) > window_size + trend_bars and bool(current_metrics.get("has_l2_coverage", False)):
        prior_end = len(bars) - trend_bars
        prior_start = max(0, prior_end - window_size)
        prior_window = bars[prior_start:prior_end]
        if len(prior_window) >= 3:
            prior_metrics = _calculate_window_flow_components(self, prior_window)
            flow_score_trend = float(current_metrics.get("flow_score", 0.0)) - float(
                prior_metrics.get("flow_score", 0.0)
            )
            signed_aggression_trend = float(current_metrics.get("signed_aggression", 0.0)) - float(
                prior_metrics.get("signed_aggression", 0.0)
            )
            book_pressure_trend_3bar = float(current_metrics.get("book_pressure_avg", 0.0)) - float(
                prior_metrics.get("book_pressure_avg", 0.0)
            )

    result = {
        **current_metrics,
        "delta_acceleration": delta_acceleration,
        "flow_score_trend_3bar": flow_score_trend,
        "signed_aggression_trend_3bar": signed_aggression_trend,
        "book_pressure_trend_3bar": book_pressure_trend_3bar,
    }
    return result
