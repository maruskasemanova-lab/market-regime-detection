"""Safe formula validation/evaluation for strategy entry/exit custom rules."""
from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Set, Tuple


MAX_FORMULA_LENGTH = 600

SUPPORTED_FORMULA_VARIABLES: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "volume",
    "vwap",
    "regime",
    "micro_regime",
    "bar_index",
    "trade_count",
    "open_positions",
    "confidence",
    "signal_side",
    "position_side",
    "bars_held",
    "position_pnl_pct",
    "position_pnl_dollars",
    "entry_price",
    "stop_loss",
    "take_profit",
    "trailing_stop_price",
    "atr",
    "adx",
    "rsi",
    "ema_fast",
    "ema_slow",
    "sma",
    "flow_score",
    "signed_aggression",
    "directional_consistency",
    "imbalance",
    "absorption_rate",
    "sweep_intensity",
    "book_pressure",
    "book_pressure_trend",
    "participation_ratio",
    "delta_zscore",
    "large_trader_activity",
    "vwap_execution_flow",
    "has_l2_coverage",
    "intrabar_move_pct",
    "intrabar_push_ratio",
    "intrabar_directional_consistency",
    "intrabar_spread_bps",
    "intrabar_micro_volatility_bps",
)


class StrategyFormulaError(ValueError):
    """Base formula error."""


class StrategyFormulaValidationError(StrategyFormulaError):
    """Raised when formula syntax/AST is not allowed."""


class StrategyFormulaEvaluationError(StrategyFormulaError):
    """Raised when formula cannot be evaluated against provided context."""


_SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Index,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)


def _normalize_formula(formula: Any) -> str:
    return str(formula or "").strip()


def _allowed_names() -> Set[str]:
    return set(SUPPORTED_FORMULA_VARIABLES) | set(_SAFE_FUNCTIONS.keys()) | {"True", "False", "None"}


def _validate_ast(tree: ast.AST) -> Set[str]:
    names_used: Set[str] = set()
    allowed_names = _allowed_names()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise StrategyFormulaValidationError(
                f"Unsupported expression element: {type(node).__name__}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise StrategyFormulaValidationError("Only direct function calls are allowed.")
            if node.func.id not in _SAFE_FUNCTIONS:
                raise StrategyFormulaValidationError(
                    f"Function '{node.func.id}' is not allowed."
                )
            if node.keywords:
                raise StrategyFormulaValidationError("Keyword arguments are not supported.")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise StrategyFormulaValidationError(
                    f"Unknown variable/function '{node.id}'."
                )
            if node.id in SUPPORTED_FORMULA_VARIABLES:
                names_used.add(node.id)
    return names_used


@lru_cache(maxsize=256)
def _compile_formula(formula: str) -> tuple[ast.AST, tuple[str, ...]]:
    normalized = _normalize_formula(formula)
    if not normalized:
        raise StrategyFormulaValidationError("Formula cannot be empty.")
    if len(normalized) > MAX_FORMULA_LENGTH:
        raise StrategyFormulaValidationError(
            f"Formula is too long (>{MAX_FORMULA_LENGTH} chars)."
        )
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise StrategyFormulaValidationError(
            f"Invalid formula syntax: {exc.msg}."
        ) from exc
    names_used = _validate_ast(tree)
    return tree, tuple(sorted(names_used))


def validate_strategy_formula(formula: Any) -> Dict[str, Any]:
    """
    Validate formula syntax/AST safety and return metadata.

    Empty formula is allowed and treated as disabled/no-op.
    """
    normalized = _normalize_formula(formula)
    if not normalized:
        return {"valid": True, "normalized": "", "variables": []}
    _, names = _compile_formula(normalized)
    return {"valid": True, "normalized": normalized, "variables": list(names)}


def _eval_node(node: ast.AST, context: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        if node.id not in context:
            raise StrategyFormulaEvaluationError(f"Missing variable '{node.id}'.")
        return context[node.id]
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value_node in node.values:
                if not bool(_eval_node(value_node, context)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value_node in node.values:
                if bool(_eval_node(value_node, context)):
                    return True
            return False
        raise StrategyFormulaEvaluationError("Unsupported boolean operator.")
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise StrategyFormulaEvaluationError("Unsupported unary operator.")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise StrategyFormulaEvaluationError("Unsupported binary operator.")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        for op, right_node in zip(node.ops, node.comparators):
            right = _eval_node(right_node, context)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise StrategyFormulaEvaluationError("Unsupported comparator.")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        fn = _eval_node(node.func, context)
        args = [_eval_node(arg, context) for arg in node.args]
        return fn(*args)
    if isinstance(node, ast.List):
        return [_eval_node(elt, context) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, context) for elt in node.elts)
    if isinstance(node, ast.Subscript):
        base = _eval_node(node.value, context)
        index = _eval_node(node.slice, context)
        return base[index]
    if isinstance(node, ast.Slice):
        lower = _eval_node(node.lower, context) if node.lower is not None else None
        upper = _eval_node(node.upper, context) if node.upper is not None else None
        step = _eval_node(node.step, context) if node.step is not None else None
        return slice(lower, upper, step)
    raise StrategyFormulaEvaluationError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_strategy_formula(
    formula: Any,
    context: Mapping[str, Any],
) -> bool:
    """Evaluate validated formula in restricted context and return boolean."""
    normalized = _normalize_formula(formula)
    if not normalized:
        return True
    try:
        tree, _ = _compile_formula(normalized)
        value = _eval_node(tree, context)
        return bool(value)
    except StrategyFormulaError:
        raise
    except Exception as exc:
        raise StrategyFormulaEvaluationError(str(exc)) from exc


def formula_variable_docs() -> Dict[str, str]:
    """Variable descriptions for UI hints/help."""
    docs = {
        "open": "Current bar open price",
        "high": "Current bar high price",
        "low": "Current bar low price",
        "close": "Current bar close price",
        "price": "Alias for close",
        "volume": "Current bar volume",
        "vwap": "Current bar/session VWAP",
        "regime": "Macro regime string (TRENDING/CHOPPY/MIXED)",
        "micro_regime": "Micro regime string",
        "bar_index": "Current bar index in session",
        "trade_count": "Number of closed trades in session",
        "open_positions": "Current open positions count",
        "confidence": "Signal confidence (entry only)",
        "signal_side": "Signal side string (buy/sell/hold)",
        "position_side": "Position side string (long/short)",
        "bars_held": "Bars since position entry",
        "position_pnl_pct": "Current position PnL in %",
        "position_pnl_dollars": "Current position PnL in USD",
        "entry_price": "Active position entry price",
        "stop_loss": "Active stop loss price",
        "take_profit": "Active take-profit price",
        "trailing_stop_price": "Active trailing stop price",
        "atr": "Latest ATR value",
        "adx": "Latest ADX value",
        "rsi": "Latest RSI value",
        "ema_fast": "Latest fast EMA value",
        "ema_slow": "Latest slow EMA value",
        "sma": "Latest SMA value",
        "flow_score": "Order-flow composite score",
        "signed_aggression": "Signed aggression metric",
        "directional_consistency": "Flow directional consistency",
        "imbalance": "Order-book imbalance average",
        "absorption_rate": "Flow absorption rate",
        "sweep_intensity": "Aggressive sweep intensity",
        "book_pressure": "Book pressure average",
        "book_pressure_trend": "Book pressure trend",
        "participation_ratio": "Aggressive participation ratio",
        "delta_zscore": "Delta z-score",
        "large_trader_activity": "Large trader activity proxy",
        "vwap_execution_flow": "VWAP execution flow proxy",
        "has_l2_coverage": "Whether L2 coverage is present",
        "intrabar_move_pct": "Intrabar move %",
        "intrabar_push_ratio": "Intrabar push ratio",
        "intrabar_directional_consistency": "Intrabar directional consistency",
        "intrabar_spread_bps": "Intrabar average spread in bps",
        "intrabar_micro_volatility_bps": "Intrabar micro-volatility in bps",
    }
    return {name: docs.get(name, "") for name in SUPPORTED_FORMULA_VARIABLES}


def formula_examples() -> Dict[str, str]:
    """Default examples to bootstrap entry/exit formula creation."""
    return {
        "entry": "regime in ('TRENDING','MIXED') and flow_score >= 55 and signed_aggression > 0.05",
        "exit": "position_side == 'long' and (position_pnl_pct < -0.35 or flow_score < 40)",
    }

