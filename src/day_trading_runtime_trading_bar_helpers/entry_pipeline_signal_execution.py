"""Signal execution/rejection helper facade for the runtime entry pipeline."""

from __future__ import annotations

from .entry_pipeline_signal_execution_core import _execute_or_reject_decision_signal

__all__ = ["_execute_or_reject_decision_signal"]
