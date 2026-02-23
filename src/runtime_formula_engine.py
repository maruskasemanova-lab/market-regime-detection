"""Safe runtime formula validation/evaluation with configurable variables."""
from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .strategy_formula_engine import (
    MAX_FORMULA_LENGTH,
    _ALLOWED_AST_NODES,
    _SAFE_FUNCTIONS,
    _eval_node,
    _normalize_formula,
    StrategyFormulaEvaluationError,
)


class RuntimeFormulaError(ValueError):
    """Base runtime-formula error."""


class RuntimeFormulaValidationError(RuntimeFormulaError):
    """Raised when runtime formula syntax/AST is not allowed."""


class RuntimeFormulaEvaluationError(RuntimeFormulaError):
    """Raised when runtime formula cannot be evaluated."""


def _allowed_names(allowed_variables: Sequence[str]) -> set[str]:
    return set(allowed_variables) | set(_SAFE_FUNCTIONS.keys()) | {"True", "False", "None"}


def _validate_ast(tree: ast.AST, *, allowed_variables: Sequence[str]) -> Tuple[str, ...]:
    names_used: set[str] = set()
    allowed_names = _allowed_names(allowed_variables)
    allowed_variable_set = set(allowed_variables)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise RuntimeFormulaValidationError(
                f"Unsupported expression element: {type(node).__name__}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise RuntimeFormulaValidationError("Only direct function calls are allowed.")
            if node.func.id not in _SAFE_FUNCTIONS:
                raise RuntimeFormulaValidationError(
                    f"Function '{node.func.id}' is not allowed."
                )
            if node.keywords:
                raise RuntimeFormulaValidationError("Keyword arguments are not supported.")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise RuntimeFormulaValidationError(
                    f"Unknown variable/function '{node.id}'."
                )
            if node.id in allowed_variable_set:
                names_used.add(node.id)
    return tuple(sorted(names_used))


@lru_cache(maxsize=1024)
def _compile_formula_cached(
    formula: str,
    allowed_variables: Tuple[str, ...],
) -> tuple[ast.AST, tuple[str, ...]]:
    normalized = _normalize_formula(formula)
    if not normalized:
        raise RuntimeFormulaValidationError("Formula cannot be empty.")
    if len(normalized) > MAX_FORMULA_LENGTH:
        raise RuntimeFormulaValidationError(
            f"Formula is too long (>{MAX_FORMULA_LENGTH} chars)."
        )
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise RuntimeFormulaValidationError(
            f"Invalid formula syntax: {exc.msg}."
        ) from exc
    names_used = _validate_ast(tree, allowed_variables=allowed_variables)
    return tree, names_used


def validate_runtime_formula(
    formula: Any,
    *,
    allowed_variables: Iterable[str],
) -> Dict[str, Any]:
    """Validate runtime formula syntax/AST safety for a hook-specific variable set."""
    normalized = _normalize_formula(formula)
    if not normalized:
        return {"valid": True, "normalized": "", "variables": []}
    allowed_tuple = tuple(sorted({str(name).strip() for name in allowed_variables if str(name).strip()}))
    _, names = _compile_formula_cached(normalized, allowed_tuple)
    return {"valid": True, "normalized": normalized, "variables": list(names)}


def evaluate_runtime_formula(
    formula: Any,
    context: Mapping[str, Any],
    *,
    allowed_variables: Iterable[str],
) -> bool:
    """Evaluate runtime formula using a hook-specific allowed-variable set."""
    normalized = _normalize_formula(formula)
    if not normalized:
        return True
    allowed_tuple = tuple(sorted({str(name).strip() for name in allowed_variables if str(name).strip()}))
    try:
        tree, _ = _compile_formula_cached(normalized, allowed_tuple)
        return bool(_eval_node(tree, context))
    except RuntimeFormulaError:
        raise
    except StrategyFormulaEvaluationError as exc:
        raise RuntimeFormulaEvaluationError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeFormulaEvaluationError(str(exc)) from exc


__all__ = [
    "RuntimeFormulaError",
    "RuntimeFormulaValidationError",
    "RuntimeFormulaEvaluationError",
    "evaluate_runtime_formula",
    "validate_runtime_formula",
]
