"""Causal market-neutral portfolio backtesting."""

from .engine import FastPortfolioConfig, performance_metrics, run_fast_backtest

__all__ = ["FastPortfolioConfig", "performance_metrics", "run_fast_backtest"]
from .attribution import attribute_pnl
from .walk_forward import WalkForwardResult, expanding_walk_forward

__all__ = ["WalkForwardResult", "attribute_pnl", "expanding_walk_forward"]
