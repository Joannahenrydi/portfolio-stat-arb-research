"""Fail-closed strategy promotion rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    orders_allowed: bool
    reasons: tuple[str, ...]


def evaluate_promotion(
    train: dict,
    validation: dict,
    *,
    minimum_sharpe: float = 0.5,
    maximum_drawdown: float = 0.20,
    paper_sessions: int = 0,
    required_paper_sessions: int = 42,
) -> PromotionDecision:
    """Require stable development evidence and a meaningful prospective paper record."""
    if not isinstance(paper_sessions, int) or not isinstance(required_paper_sessions, int):
        raise TypeError("paper session counts must be integers")
    if paper_sessions < 0 or required_paper_sessions < 1:
        raise ValueError("paper session counts must be nonnegative and requirement positive")
    reasons = []
    for name, metrics in (("train", train), ("validation", validation)):
        sharpe = metrics.get("sharpe", float("nan"))
        drawdown = metrics.get("max_drawdown", float("nan"))
        if not np.isfinite(sharpe) or sharpe < minimum_sharpe:
            reasons.append(f"{name.upper()}_SHARPE_BELOW_GATE")
        if not np.isfinite(drawdown) or drawdown < -maximum_drawdown:
            reasons.append(f"{name.upper()}_DRAWDOWN_EXCEEDS_GATE")
    if paper_sessions < required_paper_sessions:
        reasons.append("INSUFFICIENT_PROSPECTIVE_PAPER_SESSIONS")
    return PromotionDecision("ACCEPTED" if not reasons else "REJECTED", not reasons, tuple(reasons))
