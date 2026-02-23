"""L2 flow feature computation helpers for FeatureStore."""

from __future__ import annotations

import math
from typing import Any, Dict


def empty_l2_features() -> Dict[str, Any]:
    """Return empty L2 features when insufficient data."""
    return {
        'l2_has_coverage': False,
        'l2_delta': 0.0,
        'l2_signed_aggression': 0.0,
        'l2_directional_consistency': 0.0,
        'l2_imbalance': 0.0,
        'l2_absorption_rate': 0.0,
        'l2_sweep_intensity': 0.0,
        'l2_book_pressure': 0.0,
        'l2_large_trader_activity': 0.0,
        'l2_delta_zscore': 0.0,
        'l2_flow_score': 0.0,
        'l2_iceberg_bias': 0.0,
        'l2_participation_ratio': 0.0,
        'l2_delta_acceleration': 0.0,
        'l2_delta_price_divergence': 0.0,
        'l2_delta_z': 0.0,
        'l2_aggression_z': 0.0,
        'l2_imbalance_z': 0.0,
        'l2_book_pressure_z': 0.0,
        'l2_sweep_z': 0.0,
        'l2_flow_score_z': 0.0,
    }


def compute_l2_features(store: Any, bar: Dict[str, Any]) -> Dict[str, Any]:
    """Compute L2 flow features with z-score normalization."""
    l2_delta = store._to_float(bar.get('l2_delta'))
    l2_vol = max(0.0, store._to_float(bar.get('l2_volume')))
    l2_imbalance = store._to_float(bar.get('l2_imbalance'))
    l2_book_pressure = store._to_float(bar.get('l2_book_pressure'))
    l2_book_pressure_change = store._to_float(bar.get('l2_book_pressure_change'))
    l2_iceberg_bias = store._to_float(bar.get('l2_iceberg_bias'))
    l2_bid_depth = max(0.0, store._to_float(bar.get('l2_bid_depth_total')))
    l2_ask_depth = max(0.0, store._to_float(bar.get('l2_ask_depth_total')))
    bar_vol = max(0.0, float(bar.get('volume', 0)))

    has_l2 = any(bar.get(k) is not None for k in [
        'l2_delta',
        'l2_imbalance',
        'l2_volume',
        'l2_book_pressure',
        'l2_bid_depth_total',
    ])

    store._l2_deltas.append(l2_delta)
    store._l2_volumes.append(l2_vol)
    store._l2_bar_volumes.append(bar_vol)
    store._l2_imbalances.append(l2_imbalance)
    store._l2_book_pressures.append(l2_book_pressure)
    store._l2_book_pressure_changes.append(l2_book_pressure_change)
    store._l2_iceberg_biases.append(l2_iceberg_bias)
    store._l2_has_data.append(has_l2)
    store._l2_bid_depths.append(l2_bid_depth)
    store._l2_ask_depths.append(l2_ask_depth)

    lookback = 20
    window = min(lookback, len(store._l2_deltas))
    if window < 3:
        return empty_l2_features()

    deltas = list(store._l2_deltas)[-window:]
    l2_vols = list(store._l2_volumes)[-window:]
    bar_vols = list(store._l2_bar_volumes)[-window:]
    imbalances = list(store._l2_imbalances)[-window:]
    pressures = list(store._l2_book_pressures)[-window:]
    iceberg_biases = list(store._l2_iceberg_biases)[-window:]
    has_data_flags = list(store._l2_has_data)[-window:]

    bars_with_l2 = sum(1 for value in has_data_flags if value)
    has_coverage = bars_with_l2 >= max(3, window // 2)

    cum_delta = sum(deltas)
    total_l2_vol = sum(l2_vols)
    total_bar_vol = sum(bar_vols)

    signed_agg = store._safe_div(cum_delta, total_l2_vol)
    participation = store._safe_div(total_l2_vol, total_bar_vol)
    avg_imbalance = sum(imbalances) / len(imbalances) if imbalances else 0.0
    avg_pressure = sum(pressures) / len(pressures) if pressures else 0.0
    iceberg_bias_sum = sum(iceberg_biases)

    closes_list = list(store._closes)
    vwaps_list = list(store._vwaps)
    dir_hits = 0
    dir_base = 0
    absorbed_vol = 0.0
    vwap_cluster_l2_volume = 0.0
    start_idx = max(0, len(closes_list) - window)
    for i in range(window):
        ci = start_idx + i
        if ci >= len(closes_list) or ci >= len(vwaps_list):
            continue
        close_i = closes_list[ci]
        vwap_i = vwaps_list[ci]
        if close_i > 0 and vwap_i > 0:
            vwap_distance_pct = abs((close_i - vwap_i) / close_i) * 100.0
            if vwap_distance_pct <= 0.05:
                vwap_cluster_l2_volume += l2_vols[i]
        if ci < 1:
            continue
        price_chg = closes_list[ci] - closes_list[ci - 1]
        delta = deltas[i]
        if abs(delta) > 1e-9 and abs(price_chg) > 1e-9:
            dir_base += 1
            if delta * price_chg > 0:
                dir_hits += 1
        if abs(price_chg) < closes_list[ci] * 0.0002:
            absorbed_vol += l2_vols[i]

    consistency = store._safe_div(float(dir_hits), float(dir_base))
    absorption = store._safe_div(absorbed_vol, total_l2_vol)

    abs_deltas = [abs(delta) for delta in deltas]
    mean_abs_d = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    d_var = sum((ad - mean_abs_d) ** 2 for ad in abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    d_std = math.sqrt(d_var) if d_var > 0 else 0.0
    avg_l2 = sum(l2_vols) / len(l2_vols) if l2_vols else 0.0
    sweep_hits = sum(
        1
        for delta, lv in zip(abs_deltas, l2_vols)
        if delta >= (mean_abs_d + d_std) and lv >= avg_l2 * 1.2
    )
    sweep = store._safe_div(float(sweep_hits), float(window))
    vol_var = sum((lv - avg_l2) ** 2 for lv in l2_vols) / len(l2_vols) if l2_vols else 0.0
    vol_std = math.sqrt(vol_var) if vol_var > 0 else 0.0
    large_trade_threshold = avg_l2 + vol_std
    large_hits = sum(1 for lv in l2_vols if lv >= large_trade_threshold)
    large_trader = store._safe_div(float(large_hits), float(window))

    d_mean = sum(deltas) / len(deltas)
    d_variance = sum((delta - d_mean) ** 2 for delta in deltas) / len(deltas)
    d_sigma = math.sqrt(d_variance) if d_variance > 0 else 0.0
    delta_zscore = store._safe_div(deltas[-1] - d_mean, d_sigma)

    prev_start = max(0, len(store._l2_deltas) - window * 2)
    prev_end = max(0, len(store._l2_deltas) - window)
    prev_deltas = list(store._l2_deltas)[prev_start:prev_end]
    delta_accel = cum_delta - sum(prev_deltas) if prev_deltas else 0.0

    first_c = closes_list[start_idx] if start_idx < len(closes_list) else 0
    last_c = closes_list[-1] if closes_list else 0
    price_chg_pct = store._safe_div((last_c - first_c) * 100, first_c)
    rets = []
    for i in range(start_idx + 1, len(closes_list)):
        if closes_list[i - 1] > 0:
            rets.append((closes_list[i] - closes_list[i - 1]) / closes_list[i - 1] * 100)
    rv = 0.0
    if rets:
        mr = sum(rets) / len(rets)
        rv = math.sqrt(sum((r - mr) ** 2 for r in rets) / len(rets))
    vol_floor = max(rv, 0.05)
    divergence = signed_agg - (price_chg_pct / vol_floor)

    delta_efficiency = store._safe_div(
        abs(cum_delta),
        sum(abs(delta) for delta in deltas),
        0.0,
    )
    vwap_execution_flow = store._safe_div(vwap_cluster_l2_volume, total_l2_vol, 0.0)

    flow_score = 100.0 * (
        0.22 * max(0.0, min(1.0, delta_efficiency))
        + 0.19 * max(0.0, min(1.0, consistency))
        + 0.15 * max(0.0, min(1.0, abs(avg_imbalance)))
        + 0.11 * max(0.0, min(1.0, sweep))
        + 0.10 * max(0.0, min(1.0, participation))
        + 0.08 * max(0.0, min(1.0, large_trader))
        + 0.07 * max(0.0, min(1.0, vwap_execution_flow))
        + 0.08 * max(0.0, min(1.0, abs(avg_pressure)))
    )

    store._stats_l2_delta.update(cum_delta)
    store._stats_l2_aggression.update(signed_agg)
    store._stats_l2_imbalance.update(avg_imbalance)
    store._stats_l2_book_pressure.update(avg_pressure)
    store._stats_l2_sweep.update(sweep)
    store._stats_l2_flow_score.update(flow_score)

    return {
        'l2_has_coverage': has_coverage,
        'l2_delta': cum_delta,
        'l2_signed_aggression': signed_agg,
        'l2_directional_consistency': consistency,
        'l2_imbalance': avg_imbalance,
        'l2_absorption_rate': absorption,
        'l2_sweep_intensity': sweep,
        'l2_book_pressure': avg_pressure,
        'l2_large_trader_activity': large_trader,
        'l2_delta_zscore': delta_zscore,
        'l2_flow_score': flow_score,
        'l2_iceberg_bias': iceberg_bias_sum,
        'l2_participation_ratio': participation,
        'l2_delta_acceleration': delta_accel,
        'l2_delta_price_divergence': divergence,
        'l2_delta_z': store._stats_l2_delta.z_score(cum_delta),
        'l2_aggression_z': store._stats_l2_aggression.z_score(signed_agg),
        'l2_imbalance_z': store._stats_l2_imbalance.z_score(avg_imbalance),
        'l2_book_pressure_z': store._stats_l2_book_pressure.z_score(avg_pressure),
        'l2_sweep_z': store._stats_l2_sweep.z_score(sweep),
        'l2_flow_score_z': store._stats_l2_flow_score.z_score(flow_score),
    }


__all__ = ["compute_l2_features", "empty_l2_features"]
