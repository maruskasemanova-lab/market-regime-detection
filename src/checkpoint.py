"""
Checkpoint persistence for TradingOrchestrator learning state.

Serializes/deserializes: ConfidenceCalibrator, EdgeMonitor, AdaptiveWeightCombiner
to/from a JSON file for warm-start in backtest and live modes.

The checkpoint captures *learning state* (trade outcomes, calibration curves,
adaptive weights) — NOT per-session state (feature store, regime detector).
Per-session state resets at each trading day boundary via new_session().

JSON format with atomic writes (.tmp + rename) for crash safety.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .confidence_calibrator import ConfidenceCalibrator, IsotonicCalibrator
from .edge_monitor import EdgeMonitor, StrategyEdgeTracker, ROLLING_WINDOW
from .ensemble_combiner import AdaptiveWeightCombiner, SourcePerformance

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")

# Separator for tuple keys in JSON (strategy|regime, source_type|name|regime)
SEP = "|"


# ── Serialization helpers ─────────────────────────────────────────

def _serialize_isotonic(cal: IsotonicCalibrator) -> Dict[str, Any]:
    return {
        "n_bins": cal.n_bins,
        "lookback": cal.lookback,
        "outcomes": [[conf, won] for conf, won in cal._outcomes],
    }


def _serialize_calibrator(calibrator: ConfidenceCalibrator) -> Dict[str, Any]:
    cals = {}
    for (strategy, regime), iso in calibrator._calibrators.items():
        key = f"{strategy}{SEP}{regime}"
        cals[key] = _serialize_isotonic(iso)
    return {
        "lookback": calibrator._lookback,
        "calibrators": cals,
        "global_calibrator": _serialize_isotonic(calibrator._global_calibrator),
    }


def _serialize_edge_tracker(tracker: StrategyEdgeTracker) -> Dict[str, Any]:
    return {
        "window": tracker._window,
        "trades": [[pnl, won] for pnl, won in tracker._trades],
        "total_trades": tracker._total_trades,
        "total_wins": tracker._total_wins,
    }


def _serialize_edge_monitor(monitor: EdgeMonitor) -> Dict[str, Any]:
    trackers = {}
    for (strategy, regime), tracker in monitor._trackers.items():
        key = f"{strategy}{SEP}{regime}"
        trackers[key] = _serialize_edge_tracker(tracker)
    return {
        "trackers": trackers,
        "global_tracker": _serialize_edge_tracker(monitor._global_tracker),
    }


def _serialize_source_perf(perf: SourcePerformance) -> Dict[str, Any]:
    return {
        "trades": perf.trades,
        "wins": perf.wins,
        "total_pnl_r": perf.total_pnl_r,
        "recent_pnls": list(perf.recent_pnls),
    }


def _serialize_combiner(combiner: AdaptiveWeightCombiner) -> Dict[str, Any]:
    perf = {}
    for (stype, sname, regime), sp in combiner._performance.items():
        key = f"{stype}{SEP}{sname}{SEP}{regime}"
        perf[key] = _serialize_source_perf(sp)
    regime_perf = {}
    for regime, sp in combiner._regime_performance.items():
        regime_perf[regime] = _serialize_source_perf(sp)
    return {
        "performance": perf,
        "regime_performance": regime_perf,
    }


# ── Deserialization helpers ───────────────────────────────────────

def _restore_isotonic(data: Dict[str, Any]) -> IsotonicCalibrator:
    n_bins = data.get("n_bins", 10)
    lookback = data.get("lookback", 50)
    cal = IsotonicCalibrator(n_bins=n_bins, lookback=lookback)
    for conf, won in data.get("outcomes", []):
        cal._outcomes.append((float(conf), bool(won)))
    return cal


def restore_calibrator(calibrator: ConfidenceCalibrator, data: Dict[str, Any]):
    """Restore calibrator state from checkpoint data."""
    calibrator._calibrators.clear()
    calibrator._lookback = data.get("lookback", calibrator._lookback)

    for key, iso_data in data.get("calibrators", {}).items():
        parts = key.split(SEP, 1)
        if len(parts) != 2:
            logger.warning("Skipping malformed calibrator key: %s", key)
            continue
        strategy, regime = parts
        calibrator._calibrators[(strategy, regime)] = _restore_isotonic(iso_data)

    if "global_calibrator" in data:
        calibrator._global_calibrator = _restore_isotonic(data["global_calibrator"])


def restore_edge_monitor(monitor: EdgeMonitor, data: Dict[str, Any]):
    """Restore edge monitor state from checkpoint data."""
    monitor._trackers.clear()
    monitor._degradation_events.clear()

    for key, tracker_data in data.get("trackers", {}).items():
        parts = key.split(SEP, 1)
        if len(parts) != 2:
            logger.warning("Skipping malformed edge tracker key: %s", key)
            continue
        strategy, regime = parts
        window = tracker_data.get("window", ROLLING_WINDOW)
        tracker = StrategyEdgeTracker(window=window)
        for pnl, won in tracker_data.get("trades", []):
            tracker._trades.append((float(pnl), bool(won)))
        tracker._total_trades = tracker_data.get("total_trades", len(tracker._trades))
        tracker._total_wins = tracker_data.get("total_wins", 0)
        monitor._trackers[(strategy, regime)] = tracker

    if "global_tracker" in data:
        gd = data["global_tracker"]
        window = gd.get("window", ROLLING_WINDOW * 3)
        gt = StrategyEdgeTracker(window=window)
        for pnl, won in gd.get("trades", []):
            gt._trades.append((float(pnl), bool(won)))
        gt._total_trades = gd.get("total_trades", len(gt._trades))
        gt._total_wins = gd.get("total_wins", 0)
        monitor._global_tracker = gt


def _restore_source_perf(data: Dict[str, Any]) -> SourcePerformance:
    sp = SourcePerformance()
    sp.trades = data.get("trades", 0)
    sp.wins = data.get("wins", 0)
    sp.total_pnl_r = data.get("total_pnl_r", 0.0)
    for pnl in data.get("recent_pnls", []):
        sp.recent_pnls.append(float(pnl))
    return sp


def restore_combiner(combiner: AdaptiveWeightCombiner, data: Dict[str, Any]):
    """Restore combiner state from checkpoint data."""
    combiner._performance.clear()
    combiner._regime_performance.clear()

    for key, perf_data in data.get("performance", {}).items():
        parts = key.split(SEP, 2)
        if len(parts) != 3:
            logger.warning("Skipping malformed combiner key: %s", key)
            continue
        stype, sname, regime = parts
        combiner._performance[(stype, sname, regime)] = _restore_source_perf(perf_data)

    for regime, perf_data in data.get("regime_performance", {}).items():
        combiner._regime_performance[regime] = _restore_source_perf(perf_data)


# ── Public API ────────────────────────────────────────────────────

def save_checkpoint(
    calibrator: ConfidenceCalibrator,
    edge_monitor: EdgeMonitor,
    combiner: AdaptiveWeightCombiner,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save learning state to a JSON checkpoint file.

    Uses atomic write (.tmp + rename) for crash safety.
    Returns the path written.
    """
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "source": metadata or {},
        "calibrator": _serialize_calibrator(calibrator),
        "edge_monitor": _serialize_edge_monitor(edge_monitor),
        "combiner": _serialize_combiner(combiner),
    }

    # Compute summary stats for the source metadata
    total_trades = edge_monitor._global_tracker._total_trades
    total_wins = edge_monitor._global_tracker._total_wins
    checkpoint["source"]["total_trades"] = total_trades
    checkpoint["source"]["win_rate"] = (
        round(total_wins / total_trades, 3) if total_trades > 0 else 0.0
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    # Atomic rename
    os.replace(str(tmp_path), str(out))
    logger.info(
        "Checkpoint saved: %s (%d trades, %.1f%% WR)",
        out, total_trades,
        checkpoint["source"]["win_rate"] * 100,
    )
    return out


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint file. Returns the parsed dict.

    Raises FileNotFoundError if path doesn't exist.
    Raises ValueError on version mismatch or invalid format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(p) as f:
        data = json.load(f)

    version = data.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected {CHECKPOINT_VERSION}, got {version}"
        )

    return data


def apply_checkpoint(orchestrator, path: str) -> Dict[str, Any]:
    """
    Load checkpoint and restore all three learning components.

    Args:
        orchestrator: TradingOrchestrator instance
        path: Path to checkpoint JSON file

    Returns:
        Checkpoint metadata (source info, trade counts)
    """
    data = load_checkpoint(path)

    restore_calibrator(orchestrator.calibrator, data.get("calibrator", {}))
    restore_edge_monitor(orchestrator.edge_monitor, data.get("edge_monitor", {}))
    restore_combiner(orchestrator.combiner, data.get("combiner", {}))

    source = data.get("source", {})
    logger.info(
        "Checkpoint loaded: %s (trades=%d, WR=%.1f%%)",
        path,
        source.get("total_trades", 0),
        source.get("win_rate", 0) * 100,
    )
    return {
        "version": data.get("version"),
        "created_at": data.get("created_at"),
        "source": source,
    }
