"""
Streaming Feature Store with Rolling Normalization.

Centralized feature computation for bar-by-bar processing.
Replaces scattered indicator calculations with a single source of truth.
All features are normalized (z-scores / percentile ranks) to eliminate
hardcoded absolute thresholds.

Anti-bias: Rolling windows prevent look-ahead; z-score normalization
makes features context-relative rather than absolute.
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Immutable snapshot of all features at a point in time."""
    bar_index: int = 0

    # --- Raw price indicators ---
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    vwap: float = 0.0

    # --- Technical indicators (raw) ---
    sma_20: float = 0.0
    ema_10: float = 0.0
    ema_20: float = 0.0
    rsi_14: float = 50.0
    atr_14: float = 0.0
    adx_14: float = 0.0
    bollinger_width: float = 0.0
    roc_5: float = 0.0
    roc_10: float = 0.0
    roc_20: float = 0.0
    obv: float = 0.0
    obv_slope: float = 0.0
    vwap_distance_pct: float = 0.0
    range_atr_ratio: float = 0.0
    trend_efficiency: float = 0.0

    # --- Normalized (z-score) indicators ---
    rsi_z: float = 0.0
    atr_z: float = 0.0
    adx_z: float = 0.0
    volume_z: float = 0.0
    vwap_dist_z: float = 0.0
    roc_5_z: float = 0.0
    roc_10_z: float = 0.0
    momentum_z: float = 0.0  # composite momentum z-score

    # --- Percentile ranks (0-1) ---
    volume_pct_rank: float = 0.5
    atr_pct_rank: float = 0.5
    range_pct_rank: float = 0.5

    # --- L2 Flow features (raw) ---
    l2_has_coverage: bool = False
    l2_delta: float = 0.0
    l2_signed_aggression: float = 0.0
    l2_directional_consistency: float = 0.0
    l2_imbalance: float = 0.0
    l2_absorption_rate: float = 0.0
    l2_sweep_intensity: float = 0.0
    l2_book_pressure: float = 0.0
    l2_large_trader_activity: float = 0.0
    l2_delta_zscore: float = 0.0
    l2_flow_score: float = 0.0
    l2_iceberg_bias: float = 0.0
    l2_participation_ratio: float = 0.0
    l2_delta_acceleration: float = 0.0
    l2_delta_price_divergence: float = 0.0

    # --- L2 Normalized (z-score over rolling window) ---
    l2_delta_z: float = 0.0
    l2_aggression_z: float = 0.0
    l2_imbalance_z: float = 0.0
    l2_book_pressure_z: float = 0.0
    l2_sweep_z: float = 0.0
    l2_flow_score_z: float = 0.0

    # --- Multi-timeframe (aggregated from 1-min bars) ---
    tf5_trend_slope: float = 0.0   # 5-min EMA slope
    tf5_rsi: float = 50.0          # 5-min RSI
    tf15_trend_slope: float = 0.0  # 15-min EMA slope
    tf15_rsi: float = 50.0         # 15-min RSI
    tf5_volume_ratio: float = 1.0  # current 5-min vol vs avg
    tf15_volume_ratio: float = 1.0

    # --- Cross-asset (populated externally) ---
    index_trend: float = 0.0       # QQQ 5-bar momentum
    sector_relative: float = 0.0   # ticker relative strength vs sector
    correlation_20: float = 0.0    # 20-bar correlation with index
    headwind_score: float = 0.0    # 0-1, 1 = strong headwind

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for API transport."""
        return {k: v for k, v in self.__dict__.items()}


class RollingStats:
    """Efficient rolling statistics (mean, std, percentile rank)."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def update(self, value: float):
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._values) / n
        return math.sqrt(variance) if variance > 0 else 0.0

    def z_score(self, value: float) -> float:
        """Z-score of value relative to rolling window."""
        s = self.std
        if s < 1e-10:
            return 0.0
        return (value - self.mean) / s

    def percentile_rank(self, value: float) -> float:
        """Fraction of window values <= value (0 to 1)."""
        if not self._values:
            return 0.5
        count_below = sum(1 for v in self._values if v <= value)
        return count_below / len(self._values)


class FeatureStore:
    """
    Streaming feature store: accepts bars one at a time,
    maintains rolling state, outputs normalized feature vectors.

    Usage:
        store = FeatureStore()
        for bar in bars:
            fv = store.update(bar)
            # fv is a FeatureVector with all raw + normalized features
    """

    def __init__(
        self,
        zscore_window: int = 100,
        l2_zscore_window: int = 50,
        percentile_window: int = 200,
    ):
        self._bar_index = 0

        # --- Raw bar storage (for lookback calculations) ---
        self._closes: deque = deque(maxlen=500)
        self._highs: deque = deque(maxlen=500)
        self._lows: deque = deque(maxlen=500)
        self._opens: deque = deque(maxlen=500)
        self._volumes: deque = deque(maxlen=500)
        self._vwaps: deque = deque(maxlen=500)

        # --- EMA state ---
        self._ema_10: float = 0.0
        self._ema_20: float = 0.0
        self._ema_10_initialized = False
        self._ema_20_initialized = False

        # --- RSI state ---
        self._rsi_avg_gain: float = 0.0
        self._rsi_avg_loss: float = 0.0
        self._rsi_initialized = False
        self._rsi_count = 0

        # --- ATR state ---
        self._atr_values: deque = deque(maxlen=14)
        self._prev_close: Optional[float] = None

        # --- ADX state (Wilder's smoothing) ---
        self._adx_period = 14
        self._adx_tr_sum: float = 0.0
        self._adx_plus_dm_sum: float = 0.0
        self._adx_minus_dm_sum: float = 0.0
        self._adx_dx_values: deque = deque(maxlen=14)
        self._adx_smoothed: Optional[float] = None
        self._adx_bar_count = 0

        # --- OBV state ---
        self._obv: float = 0.0
        self._obv_prev_values: deque = deque(maxlen=20)

        # --- VWAP state (cumulative session) ---
        self._cum_pv: float = 0.0
        self._cum_vol: float = 0.0

        # --- Rolling stats for z-score normalization ---
        self._stats_rsi = RollingStats(zscore_window)
        self._stats_atr = RollingStats(zscore_window)
        self._stats_adx = RollingStats(zscore_window)
        self._stats_volume = RollingStats(zscore_window)
        self._stats_vwap_dist = RollingStats(zscore_window)
        self._stats_roc5 = RollingStats(zscore_window)
        self._stats_roc10 = RollingStats(zscore_window)

        # --- Percentile rank stats ---
        self._pstats_volume = RollingStats(percentile_window)
        self._pstats_atr = RollingStats(percentile_window)
        self._pstats_range = RollingStats(percentile_window)

        # --- L2 rolling stats ---
        self._stats_l2_delta = RollingStats(l2_zscore_window)
        self._stats_l2_aggression = RollingStats(l2_zscore_window)
        self._stats_l2_imbalance = RollingStats(l2_zscore_window)
        self._stats_l2_book_pressure = RollingStats(l2_zscore_window)
        self._stats_l2_sweep = RollingStats(l2_zscore_window)
        self._stats_l2_flow_score = RollingStats(l2_zscore_window)

        # --- Multi-timeframe aggregation ---
        self._tf5_bars: deque = deque(maxlen=100)  # 5-min aggregated bars
        self._tf15_bars: deque = deque(maxlen=40)   # 15-min aggregated bars
        self._tf5_accumulator: List[Dict] = []
        self._tf15_accumulator: List[Dict] = []

        # --- L2 raw values for flow metric computation ---
        self._l2_deltas: deque = deque(maxlen=50)
        self._l2_volumes: deque = deque(maxlen=50)
        self._l2_bar_volumes: deque = deque(maxlen=50)
        self._l2_imbalances: deque = deque(maxlen=50)
        self._l2_book_pressures: deque = deque(maxlen=50)
        self._l2_book_pressure_changes: deque = deque(maxlen=50)
        self._l2_iceberg_biases: deque = deque(maxlen=50)
        self._l2_has_data: deque = deque(maxlen=50)
        self._l2_bid_depths: deque = deque(maxlen=50)
        self._l2_ask_depths: deque = deque(maxlen=50)

    def reset(self):
        """Reset all state for a new session."""
        self.__init__(
            zscore_window=self._stats_rsi.window,
            l2_zscore_window=self._stats_l2_delta.window,
            percentile_window=self._pstats_volume.window,
        )

    def update(self, bar: Dict[str, Any]) -> FeatureVector:
        """
        Process a new bar and return the complete feature vector.

        Args:
            bar: dict with keys: open, high, low, close, volume, vwap (optional),
                 l2_delta, l2_imbalance, l2_volume, l2_book_pressure, etc. (optional)
        """
        self._bar_index += 1

        o = float(bar.get('open', 0))
        h = float(bar.get('high', 0))
        lo = float(bar.get('low', 0))
        c = float(bar.get('close', 0))
        v = float(bar.get('volume', 0))
        vwap = bar.get('vwap')
        if vwap is not None:
            vwap = float(vwap)

        self._closes.append(c)
        self._highs.append(h)
        self._lows.append(lo)
        self._opens.append(o)
        self._volumes.append(v)

        # --- Compute raw indicators ---
        sma_20 = self._compute_sma(20)
        ema_10 = self._compute_ema(c, 10)
        ema_20 = self._compute_ema_slow(c, 20)
        rsi_14 = self._compute_rsi(c)
        atr_14 = self._compute_atr(h, lo, c)
        adx_14 = self._compute_adx(h, lo, c)
        bollinger_width = self._compute_bollinger_width(20)
        roc_5 = self._compute_roc(5)
        roc_10 = self._compute_roc(10)
        roc_20 = self._compute_roc(20)
        obv, obv_slope = self._compute_obv(c, v)

        # VWAP
        computed_vwap = self._compute_vwap(h, lo, c, v, vwap)
        self._vwaps.append(computed_vwap)
        vwap_dist_pct = ((c - computed_vwap) / computed_vwap * 100) if computed_vwap > 0 else 0.0

        bar_range = h - lo
        range_atr = (bar_range / atr_14) if atr_14 > 0 else 0.0
        trend_eff = (abs(c - o) / bar_range) if bar_range > 0 else 0.0

        # --- Update rolling stats & compute z-scores ---
        self._stats_rsi.update(rsi_14)
        self._stats_atr.update(atr_14)
        self._stats_adx.update(adx_14)
        self._stats_volume.update(v)
        self._stats_vwap_dist.update(vwap_dist_pct)
        self._stats_roc5.update(roc_5)
        self._stats_roc10.update(roc_10)

        self._pstats_volume.update(v)
        self._pstats_atr.update(atr_14)
        self._pstats_range.update(bar_range)

        rsi_z = self._stats_rsi.z_score(rsi_14)
        atr_z = self._stats_atr.z_score(atr_14)
        adx_z = self._stats_adx.z_score(adx_14)
        volume_z = self._stats_volume.z_score(v)
        vwap_dist_z = self._stats_vwap_dist.z_score(vwap_dist_pct)
        roc_5_z = self._stats_roc5.z_score(roc_5)
        roc_10_z = self._stats_roc10.z_score(roc_10)
        momentum_z = (roc_5_z * 0.5 + roc_10_z * 0.3 + adx_z * 0.2)

        volume_pct_rank = self._pstats_volume.percentile_rank(v)
        atr_pct_rank = self._pstats_atr.percentile_rank(atr_14)
        range_pct_rank = self._pstats_range.percentile_rank(bar_range)

        # --- L2 features ---
        l2_features = self._compute_l2_features(bar)

        # --- Multi-timeframe ---
        tf_features = self._compute_multi_timeframe(bar)

        self._prev_close = c

        return FeatureVector(
            bar_index=self._bar_index,
            close=c, open=o, high=h, low=lo, volume=v, vwap=computed_vwap,
            # Raw indicators
            sma_20=sma_20, ema_10=ema_10, ema_20=ema_20,
            rsi_14=rsi_14, atr_14=atr_14, adx_14=adx_14,
            bollinger_width=bollinger_width,
            roc_5=roc_5, roc_10=roc_10, roc_20=roc_20,
            obv=obv, obv_slope=obv_slope,
            vwap_distance_pct=vwap_dist_pct,
            range_atr_ratio=range_atr, trend_efficiency=trend_eff,
            # Normalized
            rsi_z=rsi_z, atr_z=atr_z, adx_z=adx_z,
            volume_z=volume_z, vwap_dist_z=vwap_dist_z,
            roc_5_z=roc_5_z, roc_10_z=roc_10_z, momentum_z=momentum_z,
            # Percentile ranks
            volume_pct_rank=volume_pct_rank,
            atr_pct_rank=atr_pct_rank,
            range_pct_rank=range_pct_rank,
            # L2
            **l2_features,
            # Multi-timeframe
            **tf_features,
        )

    def to_legacy_indicators(self, fv: FeatureVector, order_flow: Dict) -> Dict[str, Any]:
        """
        Convert FeatureVector to legacy indicators dict format
        for backward compatibility with existing strategies.

        Returns dict with keys: sma, ema, ema_fast, ema_slow, rsi, atr, vwap, adx, order_flow
        """
        n = len(self._closes)
        return {
            'sma': list(self._closes)[-20:] if n >= 20 else list(self._closes),
            'ema': fv.ema_10,
            'ema_fast': fv.ema_10,
            'ema_slow': fv.ema_20,
            'rsi': fv.rsi_14,
            'atr': fv.atr_14,
            'vwap': fv.vwap,
            'adx': fv.adx_14,
            'order_flow': order_flow,
            # Extended features for new consumers
            '_feature_vector': fv,
        }

    # ---------------------------------------------------------------
    # Private: indicator computation
    # ---------------------------------------------------------------

    def _compute_sma(self, period: int) -> float:
        if len(self._closes) < period:
            return sum(self._closes) / len(self._closes) if self._closes else 0.0
        return sum(list(self._closes)[-period:]) / period

    def _compute_ema(self, close: float, period: int = 10) -> float:
        """EMA(10) with streaming state."""
        multiplier = 2.0 / (period + 1)
        if not self._ema_10_initialized:
            self._ema_10 = close
            self._ema_10_initialized = True
        else:
            self._ema_10 = (close - self._ema_10) * multiplier + self._ema_10
        return self._ema_10

    def _compute_ema_slow(self, close: float, period: int = 20) -> float:
        """EMA(20) with streaming state."""
        multiplier = 2.0 / (period + 1)
        if not self._ema_20_initialized:
            self._ema_20 = close
            self._ema_20_initialized = True
        else:
            self._ema_20 = (close - self._ema_20) * multiplier + self._ema_20
        return self._ema_20

    def _compute_rsi(self, close: float) -> float:
        """Wilder's RSI(14) with streaming state."""
        period = 14
        self._rsi_count += 1
        if self._prev_close is None:
            return 50.0

        change = close - self._prev_close
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))

        if not self._rsi_initialized:
            # Accumulate initial period
            self._rsi_avg_gain += gain
            self._rsi_avg_loss += loss
            if self._rsi_count > period:
                self._rsi_avg_gain /= period
                self._rsi_avg_loss /= period
                self._rsi_initialized = True
            else:
                return 50.0
        else:
            # Wilder's smoothing
            self._rsi_avg_gain = (self._rsi_avg_gain * (period - 1) + gain) / period
            self._rsi_avg_loss = (self._rsi_avg_loss * (period - 1) + loss) / period

        if self._rsi_avg_loss < 1e-10:
            return 100.0
        rs = self._rsi_avg_gain / self._rsi_avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_atr(self, high: float, low: float, close: float) -> float:
        """ATR(14) with adaptive early-session window."""
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            tr = high - low

        self._atr_values.append(tr)
        return sum(self._atr_values) / len(self._atr_values)

    def _compute_adx(self, high: float, low: float, close: float) -> float:
        """ADX(14) with Wilder's smoothing, streaming."""
        self._adx_bar_count += 1
        period = self._adx_period

        if self._adx_bar_count < 2:
            return 0.0

        prev_h = self._highs[-2] if len(self._highs) >= 2 else high
        prev_l = self._lows[-2] if len(self._lows) >= 2 else low
        prev_c = self._prev_close or close

        # True Range
        tr = max(high - low, abs(high - prev_c), abs(low - prev_c))

        # Directional Movement
        up_move = high - prev_h
        down_move = prev_l - low
        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0

        if self._adx_bar_count <= period:
            # Accumulation phase
            self._adx_tr_sum += tr
            self._adx_plus_dm_sum += plus_dm
            self._adx_minus_dm_sum += minus_dm

            if self._adx_bar_count == period:
                # First smoothed values
                pass
            return 0.0
        elif self._adx_bar_count == period + 1:
            # First proper smoothing
            self._adx_tr_sum = self._adx_tr_sum - (self._adx_tr_sum / period) + tr
            self._adx_plus_dm_sum = self._adx_plus_dm_sum - (self._adx_plus_dm_sum / period) + plus_dm
            self._adx_minus_dm_sum = self._adx_minus_dm_sum - (self._adx_minus_dm_sum / period) + minus_dm
        else:
            # Wilder's smoothing
            self._adx_tr_sum = self._adx_tr_sum - (self._adx_tr_sum / period) + tr
            self._adx_plus_dm_sum = self._adx_plus_dm_sum - (self._adx_plus_dm_sum / period) + plus_dm
            self._adx_minus_dm_sum = self._adx_minus_dm_sum - (self._adx_minus_dm_sum / period) + minus_dm

        if self._adx_tr_sum < 1e-10:
            return 0.0

        plus_di = 100.0 * self._adx_plus_dm_sum / self._adx_tr_sum
        minus_di = 100.0 * self._adx_minus_dm_sum / self._adx_tr_sum
        di_sum = plus_di + minus_di
        if di_sum < 1e-10:
            return 0.0

        dx = 100.0 * abs(plus_di - minus_di) / di_sum
        self._adx_dx_values.append(dx)

        if self._adx_smoothed is None:
            if len(self._adx_dx_values) >= period:
                self._adx_smoothed = sum(self._adx_dx_values) / len(self._adx_dx_values)
            else:
                return sum(self._adx_dx_values) / len(self._adx_dx_values) if self._adx_dx_values else 0.0
        else:
            self._adx_smoothed = (self._adx_smoothed * (period - 1) + dx) / period

        return self._adx_smoothed or 0.0

    def _compute_bollinger_width(self, period: int = 20) -> float:
        """Bollinger Band width (upper - lower) / middle."""
        if len(self._closes) < period:
            return 0.0
        window = list(self._closes)[-period:]
        mean = sum(window) / period
        if mean < 1e-10:
            return 0.0
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance) if variance > 0 else 0.0
        upper = mean + 2 * std
        lower = mean - 2 * std
        return (upper - lower) / mean

    def _compute_roc(self, period: int) -> float:
        """Rate of Change (%)."""
        if len(self._closes) <= period:
            return 0.0
        prev = self._closes[-period - 1]
        curr = self._closes[-1]
        if prev < 1e-10:
            return 0.0
        return (curr - prev) / prev * 100.0

    def _compute_obv(self, close: float, volume: float) -> tuple:
        """On Balance Volume and its slope."""
        if self._prev_close is not None:
            if close > self._prev_close:
                self._obv += volume
            elif close < self._prev_close:
                self._obv -= volume

        self._obv_prev_values.append(self._obv)
        slope = 0.0
        if len(self._obv_prev_values) >= 5:
            recent = list(self._obv_prev_values)[-5:]
            slope = (recent[-1] - recent[0]) / 4 if len(recent) >= 2 else 0.0
        return self._obv, slope

    def _compute_vwap(self, high: float, low: float, close: float,
                      volume: float, bar_vwap: Optional[float]) -> float:
        """Session VWAP - use bar's VWAP if available, else compute."""
        if bar_vwap is not None and bar_vwap > 0:
            return bar_vwap
        typical_price = (high + low + close) / 3
        self._cum_pv += typical_price * volume
        self._cum_vol += volume
        if self._cum_vol > 0:
            return self._cum_pv / self._cum_vol
        return typical_price

    # ---------------------------------------------------------------
    # L2 feature computation
    # ---------------------------------------------------------------

    def _compute_l2_features(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Compute L2 flow features with z-score normalization."""
        # Extract L2 fields from bar
        l2_delta = self._to_float(bar.get('l2_delta'))
        l2_vol = max(0.0, self._to_float(bar.get('l2_volume')))
        l2_imbalance = self._to_float(bar.get('l2_imbalance'))
        l2_book_pressure = self._to_float(bar.get('l2_book_pressure'))
        l2_book_pressure_change = self._to_float(bar.get('l2_book_pressure_change'))
        l2_iceberg_bias = self._to_float(bar.get('l2_iceberg_bias'))
        l2_bid_depth = max(0.0, self._to_float(bar.get('l2_bid_depth_total')))
        l2_ask_depth = max(0.0, self._to_float(bar.get('l2_ask_depth_total')))
        bar_vol = max(0.0, float(bar.get('volume', 0)))

        has_l2 = any(bar.get(k) is not None for k in [
            'l2_delta', 'l2_imbalance', 'l2_volume',
            'l2_book_pressure', 'l2_bid_depth_total',
        ])

        self._l2_deltas.append(l2_delta)
        self._l2_volumes.append(l2_vol)
        self._l2_bar_volumes.append(bar_vol)
        self._l2_imbalances.append(l2_imbalance)
        self._l2_book_pressures.append(l2_book_pressure)
        self._l2_book_pressure_changes.append(l2_book_pressure_change)
        self._l2_iceberg_biases.append(l2_iceberg_bias)
        self._l2_has_data.append(has_l2)
        self._l2_bid_depths.append(l2_bid_depth)
        self._l2_ask_depths.append(l2_ask_depth)

        # Compute aggregate flow metrics over lookback
        lookback = 20
        window = min(lookback, len(self._l2_deltas))
        if window < 3:
            return self._empty_l2_features()

        deltas = list(self._l2_deltas)[-window:]
        l2_vols = list(self._l2_volumes)[-window:]
        bar_vols = list(self._l2_bar_volumes)[-window:]
        imbalances = list(self._l2_imbalances)[-window:]
        pressures = list(self._l2_book_pressures)[-window:]
        iceberg_biases = list(self._l2_iceberg_biases)[-window:]
        has_data_flags = list(self._l2_has_data)[-window:]

        bars_with_l2 = sum(1 for x in has_data_flags if x)
        has_coverage = bars_with_l2 >= max(3, window // 2)

        cum_delta = sum(deltas)
        total_l2_vol = sum(l2_vols)
        total_bar_vol = sum(bar_vols)

        signed_agg = self._safe_div(cum_delta, total_l2_vol)
        participation = self._safe_div(total_l2_vol, total_bar_vol)
        avg_imbalance = sum(imbalances) / len(imbalances) if imbalances else 0.0
        avg_pressure = sum(pressures) / len(pressures) if pressures else 0.0
        iceberg_bias_sum = sum(iceberg_biases)

        # Directional consistency
        closes_list = list(self._closes)
        dir_hits = 0
        dir_base = 0
        absorbed_vol = 0.0
        start_idx = max(0, len(closes_list) - window)
        for i in range(1, window):
            ci = start_idx + i
            if ci >= len(closes_list) or ci < 1:
                continue
            price_chg = (closes_list[ci] - closes_list[ci - 1])
            d = deltas[i]
            if abs(d) > 1e-9 and abs(price_chg) > 1e-9:
                dir_base += 1
                if d * price_chg > 0:
                    dir_hits += 1
            if abs(price_chg) < closes_list[ci] * 0.0002:
                absorbed_vol += l2_vols[i]

        consistency = self._safe_div(float(dir_hits), float(dir_base))
        absorption = self._safe_div(absorbed_vol, total_l2_vol)

        # Sweep intensity
        abs_deltas = [abs(d) for d in deltas]
        mean_abs_d = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        d_var = sum((ad - mean_abs_d) ** 2 for ad in abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        d_std = math.sqrt(d_var) if d_var > 0 else 0.0
        avg_l2 = sum(l2_vols) / len(l2_vols) if l2_vols else 0.0
        sweep_hits = sum(1 for d, lv in zip(abs_deltas, l2_vols)
                         if d >= (mean_abs_d + d_std) and lv >= avg_l2 * 1.2)
        sweep = self._safe_div(float(sweep_hits), float(window))
        large_hits = sum(1 for lv in l2_vols if lv >= max(avg_l2 * 1.8, 5000.0))
        large_trader = self._safe_div(float(large_hits), float(window))

        # Delta z-score
        d_mean = sum(deltas) / len(deltas)
        d_variance = sum((d - d_mean) ** 2 for d in deltas) / len(deltas)
        d_sigma = math.sqrt(d_variance) if d_variance > 0 else 0.0
        delta_zscore = self._safe_div(deltas[-1] - d_mean, d_sigma)

        # Delta acceleration
        prev_start = max(0, len(self._l2_deltas) - window * 2)
        prev_end = max(0, len(self._l2_deltas) - window)
        prev_deltas = list(self._l2_deltas)[prev_start:prev_end]
        delta_accel = cum_delta - sum(prev_deltas) if prev_deltas else 0.0

        # Price divergence
        first_c = closes_list[start_idx] if start_idx < len(closes_list) else 0
        last_c = closes_list[-1] if closes_list else 0
        price_chg_pct = self._safe_div((last_c - first_c) * 100, first_c)
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

        # Flow score (composite)
        flow_score = 100.0 * (
            0.22 * self._safe_div(abs(cum_delta), abs(cum_delta) + 5000.0)
            + 0.19 * max(0.0, min(1.0, consistency))
            + 0.15 * max(0.0, min(1.0, abs(avg_imbalance)))
            + 0.11 * max(0.0, min(1.0, sweep))
            + 0.10 * max(0.0, min(1.0, participation))
            + 0.08 * max(0.0, min(1.0, large_trader))
            + 0.07 * max(0.0, min(1.0, 0.0))  # vwap_execution placeholder
            + 0.08 * max(0.0, min(1.0, abs(avg_pressure)))
        )

        # Update z-score stats
        self._stats_l2_delta.update(cum_delta)
        self._stats_l2_aggression.update(signed_agg)
        self._stats_l2_imbalance.update(avg_imbalance)
        self._stats_l2_book_pressure.update(avg_pressure)
        self._stats_l2_sweep.update(sweep)
        self._stats_l2_flow_score.update(flow_score)

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
            # Normalized
            'l2_delta_z': self._stats_l2_delta.z_score(cum_delta),
            'l2_aggression_z': self._stats_l2_aggression.z_score(signed_agg),
            'l2_imbalance_z': self._stats_l2_imbalance.z_score(avg_imbalance),
            'l2_book_pressure_z': self._stats_l2_book_pressure.z_score(avg_pressure),
            'l2_sweep_z': self._stats_l2_sweep.z_score(sweep),
            'l2_flow_score_z': self._stats_l2_flow_score.z_score(flow_score),
        }

    def _empty_l2_features(self) -> Dict[str, Any]:
        """Return empty L2 features when insufficient data."""
        return {
            'l2_has_coverage': False,
            'l2_delta': 0.0, 'l2_signed_aggression': 0.0,
            'l2_directional_consistency': 0.0, 'l2_imbalance': 0.0,
            'l2_absorption_rate': 0.0, 'l2_sweep_intensity': 0.0,
            'l2_book_pressure': 0.0, 'l2_large_trader_activity': 0.0,
            'l2_delta_zscore': 0.0, 'l2_flow_score': 0.0,
            'l2_iceberg_bias': 0.0, 'l2_participation_ratio': 0.0,
            'l2_delta_acceleration': 0.0, 'l2_delta_price_divergence': 0.0,
            'l2_delta_z': 0.0, 'l2_aggression_z': 0.0,
            'l2_imbalance_z': 0.0, 'l2_book_pressure_z': 0.0,
            'l2_sweep_z': 0.0, 'l2_flow_score_z': 0.0,
        }

    # ---------------------------------------------------------------
    # Multi-timeframe aggregation
    # ---------------------------------------------------------------

    def _compute_multi_timeframe(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate 1-min bars into 5-min and 15-min timeframes."""
        bar_entry = {
            'open': float(bar.get('open', 0)),
            'high': float(bar.get('high', 0)),
            'low': float(bar.get('low', 0)),
            'close': float(bar.get('close', 0)),
            'volume': float(bar.get('volume', 0)),
        }
        self._tf5_accumulator.append(bar_entry)
        self._tf15_accumulator.append(bar_entry)

        # Aggregate to 5-min bar
        if len(self._tf5_accumulator) >= 5:
            agg = self._aggregate_bars(self._tf5_accumulator)
            self._tf5_bars.append(agg)
            self._tf5_accumulator = []

        # Aggregate to 15-min bar
        if len(self._tf15_accumulator) >= 15:
            agg = self._aggregate_bars(self._tf15_accumulator)
            self._tf15_bars.append(agg)
            self._tf15_accumulator = []

        return {
            'tf5_trend_slope': self._compute_tf_ema_slope(self._tf5_bars, 5),
            'tf5_rsi': self._compute_tf_rsi(self._tf5_bars),
            'tf15_trend_slope': self._compute_tf_ema_slope(self._tf15_bars, 5),
            'tf15_rsi': self._compute_tf_rsi(self._tf15_bars),
            'tf5_volume_ratio': self._compute_tf_volume_ratio(self._tf5_bars),
            'tf15_volume_ratio': self._compute_tf_volume_ratio(self._tf15_bars),
        }

    @staticmethod
    def _aggregate_bars(bars: List[Dict]) -> Dict:
        """Aggregate minute bars into a single higher-TF bar."""
        return {
            'open': bars[0]['open'],
            'high': max(b['high'] for b in bars),
            'low': min(b['low'] for b in bars),
            'close': bars[-1]['close'],
            'volume': sum(b['volume'] for b in bars),
        }

    @staticmethod
    def _compute_tf_ema_slope(tf_bars: deque, period: int = 5) -> float:
        """EMA slope on higher timeframe bars."""
        if len(tf_bars) < period + 1:
            return 0.0
        closes = [b['close'] for b in list(tf_bars)[-period - 1:]]
        mult = 2.0 / (period + 1)
        ema = closes[0]
        prev_ema = ema
        for c in closes[1:]:
            prev_ema = ema
            ema = (c - ema) * mult + ema
        return ema - prev_ema

    @staticmethod
    def _compute_tf_rsi(tf_bars: deque, period: int = 14) -> float:
        """Simple RSI on higher timeframe bars."""
        if len(tf_bars) < period + 1:
            return 50.0
        closes = [b['close'] for b in list(tf_bars)[-(period + 1):]]
        gains, losses = [], []
        for i in range(1, len(closes)):
            chg = closes[i] - closes[i - 1]
            gains.append(max(chg, 0))
            losses.append(abs(min(chg, 0)))
        avg_g = sum(gains) / len(gains) if gains else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        if avg_l < 1e-10:
            return 100.0
        rs = avg_g / avg_l
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_tf_volume_ratio(tf_bars: deque) -> float:
        """Current TF bar volume vs rolling average."""
        if len(tf_bars) < 2:
            return 1.0
        vols = [b['volume'] for b in tf_bars]
        avg = sum(vols[:-1]) / len(vols[:-1]) if len(vols) > 1 else 1.0
        if avg < 1e-10:
            return 1.0
        return vols[-1] / avg

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
