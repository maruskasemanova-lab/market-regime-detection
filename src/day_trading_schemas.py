from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SleeveConfig(BaseModel):
    sleeve_id: str
    enabled: bool = False
    weight: float = 1.0


class MomentumDiversificationConfig(BaseModel):
    enabled: bool = False
    gate_mode: str = "weighted"
    gate_threshold: float = 0.55
    gate_flow_floor: float = 40.0
    apply_to_strategies: List[str] = Field(default_factory=list)
    sleeves: List[SleeveConfig] = Field(default_factory=list)

    # Thresholds
    min_flow_score: float = 55.0
    min_directional_consistency: float = 0.35
    min_signed_aggression: float = 0.03
    min_imbalance: float = 0.02
    min_cvd: float = 0.0
    min_directional_price_change_pct: float = 0.0
    min_price_trend_efficiency: float = 0.0
    min_last_bar_body_ratio: float = 0.0
    min_last_bar_close_location: float = 0.0
    min_delta_acceleration: float = 0.0
    min_delta_price_divergence: float = -0.45
    
    # Coverage / Regime filtering
    require_l2_coverage: bool = True
    blocked_micro_regimes: List[str] = Field(default_factory=list)
    allowed_micro_regimes: List[str] = Field(default_factory=list)

    @field_validator('apply_to_strategies', 'blocked_micro_regimes', 'allowed_micro_regimes', mode='before')
    def split_comma_strings(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(',') if s.strip()]
        return v


class ExecutionConfig(BaseModel):
    use_twap: bool = False
    use_vwap: bool = False
    twap_intervals: int = 1
    vwap_participation_rate: float = 0.1
    max_slippage_bps: float = 5.0
    execution_timeout_seconds: float = 60.0
    stagger_entries: bool = False
    stagger_steps: int = 1


class AdaptiveConfig(BaseModel):
    regime_detection_enabled: bool = True
    macro_regime_lookback: int = 60
    micro_regime_lookback: int = 10
    
    momentum_diversification: Optional[MomentumDiversificationConfig] = None
    execution: Optional[ExecutionConfig] = None
    
    # Pass-through for legacy/untapped config
    raw: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveConfig":
        if not data:
            return cls(raw={})
        
        filtered = {k: v for k, v in data.items() if k in cls.model_fields}
        filtered["raw"] = data
        return cls.model_validate(filtered)
