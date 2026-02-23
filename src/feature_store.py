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
from collections import deque
from typing import Dict, List, Optional, Any

from .feature_store_helpers.l2 import compute_l2_features, empty_l2_features
from .feature_store_helpers.timeframe import (
    aggregate_bars,
    compute_multi_timeframe,
    compute_tf_ema_slope,
    compute_tf_rsi,
    compute_tf_volume_ratio,
)
from .feature_store_helpers.types import FeatureVector, RollingStats


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
        trend_eff = self._compute_price_trend_efficiency()

        self._stats_rsi.update(rsi_14)
        self._stats_atr.update(atr_14)
        if adx_14 is not None:
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
        adx_z = self._stats_adx.z_score(adx_14) if adx_14 is not None else 0.0
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
            'volume_pct_rank': fv.volume_pct_rank,
            'tf5_trend_slope': fv.tf5_trend_slope,
            'tf15_rsi': fv.tf15_rsi,
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

    def _compute_adx(self, high: float, low: float, close: float) -> Optional[float]:
        """ADX(14) with Wilder's smoothing, streaming. Returns None during warmup."""
        self._adx_bar_count += 1
        period = self._adx_period

        if self._adx_bar_count < 2:
            return None

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
            return None
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
            return None

        plus_di = 100.0 * self._adx_plus_dm_sum / self._adx_tr_sum
        minus_di = 100.0 * self._adx_minus_dm_sum / self._adx_tr_sum
        di_sum = plus_di + minus_di
        if di_sum < 1e-10:
            return None

        dx = 100.0 * abs(plus_di - minus_di) / di_sum
        self._adx_dx_values.append(dx)

        if self._adx_smoothed is None:
            if len(self._adx_dx_values) >= period:
                self._adx_smoothed = sum(self._adx_dx_values) / len(self._adx_dx_values)
            else:
                # Still in warmup - not enough DX values for proper smoothing
                return None
        else:
            self._adx_smoothed = (self._adx_smoothed * (period - 1) + dx) / period

        return self._adx_smoothed

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

    def _compute_price_trend_efficiency(self, lookback: int = 30) -> float:
        """
        Multi-bar directional efficiency: |net move| / total absolute move.

        This matches legacy regime logic and avoids single-bar body/range noise.
        """
        closes = list(self._closes)
        if len(closes) < 2:
            return 0.0
        window = closes[-min(max(2, int(lookback)), len(closes)):]
        net_move = abs(window[-1] - window[0])
        total_move = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
        return self._safe_div(net_move, total_move, 0.0)

    # ---------------------------------------------------------------
    # L2 feature computation
    # ---------------------------------------------------------------

    def _compute_l2_features(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        return compute_l2_features(self, bar)

    def _empty_l2_features(self) -> Dict[str, Any]:
        return empty_l2_features()

    # ---------------------------------------------------------------
    # Multi-timeframe aggregation
    # ---------------------------------------------------------------

    def _compute_multi_timeframe(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        return compute_multi_timeframe(self, bar)

    @staticmethod
    def _aggregate_bars(bars: List[Dict]) -> Dict:
        return aggregate_bars(bars)

    @staticmethod
    def _compute_tf_ema_slope(tf_bars: deque, period: int = 5) -> float:
        return compute_tf_ema_slope(tf_bars, period)

    @staticmethod
    def _compute_tf_rsi(tf_bars: deque, period: int = 14) -> float:
        return compute_tf_rsi(tf_bars, period)

    @staticmethod
    def _compute_tf_volume_ratio(tf_bars: deque) -> float:
        return compute_tf_volume_ratio(tf_bars)

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
