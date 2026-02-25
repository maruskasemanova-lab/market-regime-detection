"""BarData construction helpers for DayTradingManager intrabar flows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from ..day_trading_models import BarData


def build_intrabar_slice_bar_data(
    timestamp: datetime,
    bar_data: Mapping[str, Any],
) -> BarData:
    """Build the BarData payload used by intrabar slice evaluation."""

    return BarData(
        timestamp=timestamp,
        open=bar_data.get("open", 0),
        high=bar_data.get("high", 0),
        low=bar_data.get("low", 0),
        close=bar_data.get("close", 0),
        volume=bar_data.get("volume", 0),
        vwap=bar_data.get("vwap"),
        l2_delta=bar_data.get("l2_delta"),
        l2_buy_volume=bar_data.get("l2_buy_volume"),
        l2_sell_volume=bar_data.get("l2_sell_volume"),
        l2_volume=bar_data.get("l2_volume"),
        l2_imbalance=bar_data.get("l2_imbalance"),
        l2_bid_depth_total=bar_data.get("l2_bid_depth_total"),
        l2_ask_depth_total=bar_data.get("l2_ask_depth_total"),
        l2_book_pressure=bar_data.get("l2_book_pressure"),
        l2_book_pressure_change=bar_data.get("l2_book_pressure_change"),
        l2_iceberg_buy_count=bar_data.get("l2_iceberg_buy_count"),
        l2_iceberg_sell_count=bar_data.get("l2_iceberg_sell_count"),
        l2_iceberg_bias=bar_data.get("l2_iceberg_bias"),
        l2_quality_flags=bar_data.get("l2_quality_flags"),
        l2_quality=bar_data.get("l2_quality"),
        intrabar_quotes_1s=bar_data.get("intrabar_quotes_1s"),
    )
