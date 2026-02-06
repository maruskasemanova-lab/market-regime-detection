"""
Candlestick Pattern Detector - Layer 1 of multi-layer decision engine.
Detects candlestick patterns from OHLCV data optimized for NASDAQ stocks.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DetectedPattern:
    """A detected candlestick pattern."""
    name: str
    direction: PatternDirection
    strength: float  # 0-100
    bar_count: int  # how many bars form the pattern
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'direction': self.direction.value,
            'strength': round(self.strength, 1),
            'bar_count': self.bar_count,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
        }


class CandlestickPatternDetector:
    """
    Detects candlestick patterns from OHLCV data.
    Optimized for NASDAQ large-cap stocks (1-minute bars).

    Patterns are detected from the most recent bars in the OHLCV arrays.
    Strength is modified by volume confirmation, VWAP proximity, and trend alignment.
    """

    def __init__(
        self,
        body_doji_pct: float = 0.1,
        wick_ratio_hammer: float = 2.0,
        engulfing_min_body_pct: float = 0.3,
        volume_confirm_ratio: float = 1.3,
        vwap_proximity_pct: float = 0.3,
    ):
        self.body_doji_pct = body_doji_pct
        self.wick_ratio_hammer = wick_ratio_hammer
        self.engulfing_min_body_pct = engulfing_min_body_pct
        self.volume_confirm_ratio = volume_confirm_ratio
        self.vwap_proximity_pct = vwap_proximity_pct

    def detect(
        self,
        ohlcv: Dict[str, List[float]],
        indicators: Optional[Dict[str, Any]] = None,
    ) -> List[DetectedPattern]:
        """
        Detect all candlestick patterns from the most recent bars.

        Args:
            ohlcv: Dict with 'open', 'high', 'low', 'close', 'volume' arrays
            indicators: Optional dict with 'vwap', 'ema', 'ema_fast', etc.

        Returns:
            List of DetectedPattern objects found on the current bar.
        """
        opens = ohlcv.get('open', [])
        highs = ohlcv.get('high', [])
        lows = ohlcv.get('low', [])
        closes = ohlcv.get('close', [])
        volumes = ohlcv.get('volume', [])

        if len(closes) < 3:
            return []

        patterns: List[DetectedPattern] = []

        # Single-bar patterns (use last bar)
        patterns.extend(self._detect_hammer(opens, highs, lows, closes))
        patterns.extend(self._detect_shooting_star(opens, highs, lows, closes))
        patterns.extend(self._detect_doji(opens, highs, lows, closes))
        patterns.extend(self._detect_marubozu(opens, highs, lows, closes))

        # Two-bar patterns (use last 2 bars)
        if len(closes) >= 2:
            patterns.extend(self._detect_engulfing(opens, highs, lows, closes))
            patterns.extend(self._detect_piercing_dark_cloud(opens, highs, lows, closes))
            patterns.extend(self._detect_harami(opens, highs, lows, closes))

        # Three-bar patterns (use last 3 bars)
        if len(closes) >= 3:
            patterns.extend(self._detect_morning_evening_star(opens, highs, lows, closes))
            patterns.extend(self._detect_three_soldiers_crows(opens, highs, lows, closes))

        # Apply strength modifiers
        patterns = self._apply_modifiers(patterns, ohlcv, indicators)

        return patterns

    # ── Helpers ──────────────────────────────────────────────────────

    def _body(self, o: float, c: float) -> float:
        return abs(c - o)

    def _range(self, h: float, l: float) -> float:
        return h - l

    def _is_bullish(self, o: float, c: float) -> bool:
        return c > o

    def _is_bearish(self, o: float, c: float) -> bool:
        return c < o

    def _upper_wick(self, o: float, h: float, c: float) -> float:
        return h - max(o, c)

    def _lower_wick(self, o: float, l: float, c: float) -> float:
        return min(o, c) - l

    # ── Single-bar patterns ──────────────────────────────────────────

    def _detect_hammer(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Hammer: small body at top, long lower wick >= 2x body, tiny upper wick."""
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = self._body(o, c)
        rng = self._range(h, l)
        if rng == 0:
            return []

        lower_wick = self._lower_wick(o, l, c)
        upper_wick = self._upper_wick(o, h, c)

        body_pct = body / rng
        if body_pct > 0.35:
            return []
        if body == 0:
            return []
        if lower_wick < body * self.wick_ratio_hammer:
            return []
        if upper_wick > body * 0.5:
            return []

        # Require preceding decline (at least 2 of last 3 bars bearish)
        if len(closes) >= 4:
            bearish_count = sum(
                1 for i in range(-4, -1) if closes[i] < opens[i]
            )
            if bearish_count < 2:
                return []

        return [DetectedPattern(
            name="Hammer",
            direction=PatternDirection.BULLISH,
            strength=65.0,
            bar_count=1,
            reasoning=f"Hammer: lower wick {lower_wick/body:.1f}x body, body {body_pct*100:.0f}% of range",
        )]

    def _detect_shooting_star(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Shooting Star: small body at bottom, long upper wick >= 2x body."""
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = self._body(o, c)
        rng = self._range(h, l)
        if rng == 0:
            return []

        upper_wick = self._upper_wick(o, h, c)
        lower_wick = self._lower_wick(o, l, c)

        body_pct = body / rng
        if body_pct > 0.35:
            return []
        if body == 0:
            return []
        if upper_wick < body * self.wick_ratio_hammer:
            return []
        if lower_wick > body * 0.5:
            return []

        # Require preceding advance
        if len(closes) >= 4:
            bullish_count = sum(
                1 for i in range(-4, -1) if closes[i] > opens[i]
            )
            if bullish_count < 2:
                return []

        return [DetectedPattern(
            name="Shooting Star",
            direction=PatternDirection.BEARISH,
            strength=65.0,
            bar_count=1,
            reasoning=f"Shooting Star: upper wick {upper_wick/body:.1f}x body, body {body_pct*100:.0f}% of range",
        )]

    def _detect_doji(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Doji: very small body relative to range."""
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = self._body(o, c)
        rng = self._range(h, l)
        if rng == 0:
            return []

        body_pct = body / rng
        if body_pct > self.body_doji_pct:
            return []

        return [DetectedPattern(
            name="Doji",
            direction=PatternDirection.NEUTRAL,
            strength=40.0,
            bar_count=1,
            reasoning=f"Doji: body only {body_pct*100:.1f}% of range, indecision",
        )]

    def _detect_marubozu(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Marubozu: full-body candle with very small wicks (>90% body)."""
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = self._body(o, c)
        rng = self._range(h, l)
        if rng == 0:
            return []

        body_pct = body / rng
        if body_pct < 0.90:
            return []

        direction = PatternDirection.BULLISH if c > o else PatternDirection.BEARISH
        label = "Bullish" if direction == PatternDirection.BULLISH else "Bearish"

        return [DetectedPattern(
            name=f"{label} Marubozu",
            direction=direction,
            strength=60.0,
            bar_count=1,
            reasoning=f"{label} Marubozu: body {body_pct*100:.0f}% of range, strong momentum",
        )]

    # ── Two-bar patterns ─────────────────────────────────────────────

    def _detect_engulfing(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Bullish/Bearish Engulfing: current bar body engulfs previous bar body."""
        o1, c1 = opens[-2], closes[-2]
        o2, c2 = opens[-1], closes[-1]
        body1 = self._body(o1, c1)
        body2 = self._body(o2, c2)
        rng2 = self._range(highs[-1], lows[-1])

        if rng2 == 0 or body1 == 0:
            return []
        if body2 / rng2 < self.engulfing_min_body_pct:
            return []

        patterns = []

        # Bullish engulfing: prev bearish, current bullish, current body engulfs prev
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 <= c1 and c2 >= o1:
                patterns.append(DetectedPattern(
                    name="Bullish Engulfing",
                    direction=PatternDirection.BULLISH,
                    strength=75.0,
                    bar_count=2,
                    reasoning=f"Bullish Engulfing: current body ({body2:.2f}) engulfs prev ({body1:.2f})",
                ))

        # Bearish engulfing: prev bullish, current bearish, current body engulfs prev
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 >= c1 and c2 <= o1:
                patterns.append(DetectedPattern(
                    name="Bearish Engulfing",
                    direction=PatternDirection.BEARISH,
                    strength=75.0,
                    bar_count=2,
                    reasoning=f"Bearish Engulfing: current body ({body2:.2f}) engulfs prev ({body1:.2f})",
                ))

        return patterns

    def _detect_piercing_dark_cloud(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Piercing Line / Dark Cloud Cover: close penetrates > 50% of prev body."""
        o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
        o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1]
        body1 = self._body(o1, c1)

        if body1 == 0:
            return []

        midpoint1 = (o1 + c1) / 2
        patterns = []

        # Piercing Line: prev bearish, current opens below prev low, closes above midpoint
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 <= l1 and c2 > midpoint1 and c2 < o1:
                penetration = (c2 - c1) / body1 * 100
                patterns.append(DetectedPattern(
                    name="Piercing Line",
                    direction=PatternDirection.BULLISH,
                    strength=60.0,
                    bar_count=2,
                    reasoning=f"Piercing Line: {penetration:.0f}% penetration of prev body",
                ))

        # Dark Cloud Cover: prev bullish, current opens above prev high, closes below midpoint
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 >= h1 and c2 < midpoint1 and c2 > o1:
                penetration = (c1 - c2) / body1 * 100
                patterns.append(DetectedPattern(
                    name="Dark Cloud Cover",
                    direction=PatternDirection.BEARISH,
                    strength=60.0,
                    bar_count=2,
                    reasoning=f"Dark Cloud Cover: {penetration:.0f}% penetration of prev body",
                ))

        return patterns

    def _detect_harami(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Bullish/Bearish Harami: small body contained within previous larger body."""
        o1, c1 = opens[-2], closes[-2]
        o2, c2 = opens[-1], closes[-1]
        body1 = self._body(o1, c1)
        body2 = self._body(o2, c2)

        if body1 == 0:
            return []
        if body2 >= body1 * 0.6:
            return []

        patterns = []
        upper1, lower1 = max(o1, c1), min(o1, c1)

        # Body of current bar must be inside body of previous bar
        upper2, lower2 = max(o2, c2), min(o2, c2)
        if upper2 > upper1 or lower2 < lower1:
            return []

        # Bullish Harami: prev bearish, current bullish, small body inside
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            patterns.append(DetectedPattern(
                name="Bullish Harami",
                direction=PatternDirection.BULLISH,
                strength=55.0,
                bar_count=2,
                reasoning=f"Bullish Harami: small bullish body ({body2:.2f}) inside bearish ({body1:.2f})",
            ))

        # Bearish Harami: prev bullish, current bearish
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            patterns.append(DetectedPattern(
                name="Bearish Harami",
                direction=PatternDirection.BEARISH,
                strength=55.0,
                bar_count=2,
                reasoning=f"Bearish Harami: small bearish body ({body2:.2f}) inside bullish ({body1:.2f})",
            ))

        return patterns

    # ── Three-bar patterns ───────────────────────────────────────────

    def _detect_morning_evening_star(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Morning Star / Evening Star: three-bar reversal patterns."""
        o1, c1 = opens[-3], closes[-3]
        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        o3, c3 = opens[-1], closes[-1]

        body1 = self._body(o1, c1)
        body2 = self._body(o2, c2)
        body3 = self._body(o3, c3)
        rng2 = self._range(h2, l2)

        if body1 == 0 or body3 == 0:
            return []

        # Star: middle bar has small body
        is_star = rng2 > 0 and (body2 / rng2) < 0.4

        patterns = []

        # Morning Star: bearish, star, bullish
        if (self._is_bearish(o1, c1) and is_star and self._is_bullish(o3, c3)):
            # Third bar must close above midpoint of first bar
            mid1 = (o1 + c1) / 2
            if c3 > mid1:
                patterns.append(DetectedPattern(
                    name="Morning Star",
                    direction=PatternDirection.BULLISH,
                    strength=80.0,
                    bar_count=3,
                    reasoning="Morning Star: bearish -> star -> bullish reversal",
                ))

        # Evening Star: bullish, star, bearish
        if (self._is_bullish(o1, c1) and is_star and self._is_bearish(o3, c3)):
            mid1 = (o1 + c1) / 2
            if c3 < mid1:
                patterns.append(DetectedPattern(
                    name="Evening Star",
                    direction=PatternDirection.BEARISH,
                    strength=80.0,
                    bar_count=3,
                    reasoning="Evening Star: bullish -> star -> bearish reversal",
                ))

        return patterns

    def _detect_three_soldiers_crows(
        self, opens: list, highs: list, lows: list, closes: list
    ) -> List[DetectedPattern]:
        """Three White Soldiers / Three Black Crows."""
        patterns = []

        # Check last 3 bars
        all_bullish = all(closes[-i] > opens[-i] for i in range(1, 4))
        all_bearish = all(closes[-i] < opens[-i] for i in range(1, 4))

        if all_bullish:
            # Each bar opens within previous bar's body and closes higher
            valid = True
            for i in range(2, 0, -1):  # bars -2 and -1
                prev_o, prev_c = opens[-(i + 1)], closes[-(i + 1)]
                curr_o = opens[-i]
                # Open within previous body
                if curr_o < min(prev_o, prev_c) or curr_o > max(prev_o, prev_c):
                    valid = False
                    break

            # Progressive closes
            if valid and closes[-1] > closes[-2] > closes[-3]:
                # Check bodies are significant
                bodies = [self._body(opens[-i], closes[-i]) for i in range(1, 4)]
                ranges = [self._range(highs[-i], lows[-i]) for i in range(1, 4)]
                if all(r > 0 and b / r > 0.5 for b, r in zip(bodies, ranges)):
                    patterns.append(DetectedPattern(
                        name="Three White Soldiers",
                        direction=PatternDirection.BULLISH,
                        strength=70.0,
                        bar_count=3,
                        reasoning="Three White Soldiers: 3 consecutive bullish bars with progressive closes",
                    ))

        if all_bearish:
            valid = True
            for i in range(2, 0, -1):
                prev_o, prev_c = opens[-(i + 1)], closes[-(i + 1)]
                curr_o = opens[-i]
                if curr_o < min(prev_o, prev_c) or curr_o > max(prev_o, prev_c):
                    valid = False
                    break

            if valid and closes[-1] < closes[-2] < closes[-3]:
                bodies = [self._body(opens[-i], closes[-i]) for i in range(1, 4)]
                ranges = [self._range(highs[-i], lows[-i]) for i in range(1, 4)]
                if all(r > 0 and b / r > 0.5 for b, r in zip(bodies, ranges)):
                    patterns.append(DetectedPattern(
                        name="Three Black Crows",
                        direction=PatternDirection.BEARISH,
                        strength=70.0,
                        bar_count=3,
                        reasoning="Three Black Crows: 3 consecutive bearish bars with progressive closes",
                    ))

        return patterns

    # ── Strength modifiers ───────────────────────────────────────────

    def _apply_modifiers(
        self,
        patterns: List[DetectedPattern],
        ohlcv: Dict[str, List[float]],
        indicators: Optional[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Apply strength modifiers based on volume, VWAP, trend alignment."""
        if not patterns:
            return patterns

        volumes = ohlcv.get('volume', [])
        closes = ohlcv.get('close', [])

        # Volume confirmation
        if len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            curr_vol = volumes[-1]
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

            for p in patterns:
                if vol_ratio >= self.volume_confirm_ratio:
                    p.strength += 15
                    p.metadata['volume_confirmed'] = True
                    p.metadata['volume_ratio'] = round(vol_ratio, 2)
                else:
                    p.metadata['volume_confirmed'] = False
                    p.metadata['volume_ratio'] = round(vol_ratio, 2)

        # VWAP proximity
        if indicators:
            vwap = indicators.get('vwap')
            if vwap:
                vwap_val = vwap[-1] if isinstance(vwap, list) else vwap
                price = closes[-1] if closes else 0
                if price > 0 and vwap_val > 0:
                    dist_pct = abs(price - vwap_val) / vwap_val * 100
                    for p in patterns:
                        if dist_pct <= self.vwap_proximity_pct:
                            p.strength += 10
                            p.metadata['near_vwap'] = True
                        p.metadata['vwap_distance_pct'] = round(dist_pct, 2)

            # Trend alignment with EMA
            ema = indicators.get('ema') or indicators.get('ema_fast')
            if ema:
                ema_val = ema[-1] if isinstance(ema, list) else ema
                price = closes[-1] if closes else 0
                for p in patterns:
                    if p.direction == PatternDirection.BULLISH and price > ema_val:
                        p.strength += 10
                        p.metadata['trend_aligned'] = True
                    elif p.direction == PatternDirection.BEARISH and price < ema_val:
                        p.strength += 10
                        p.metadata['trend_aligned'] = True
                    else:
                        p.metadata['trend_aligned'] = False

        # Multiple patterns bonus
        if len(patterns) > 1:
            bullish = [p for p in patterns if p.direction == PatternDirection.BULLISH]
            bearish = [p for p in patterns if p.direction == PatternDirection.BEARISH]

            if len(bullish) > 1:
                for p in bullish:
                    p.strength += 5 * (len(bullish) - 1)
                    p.metadata['multi_pattern_bonus'] = True
            if len(bearish) > 1:
                for p in bearish:
                    p.strength += 5 * (len(bearish) - 1)
                    p.metadata['multi_pattern_bonus'] = True

        # Cap strength at 100
        for p in patterns:
            p.strength = min(p.strength, 100.0)

        return patterns
