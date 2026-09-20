import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "multi_pair", Path(__file__).resolve().parents[1] / "scripts/evaluate_multi_pair.py")
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


def test_portfolio_uses_one_cash_budget():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    summary = {"entries": 1, "commission": 2, "slippage": 3, "borrow": 4}
    sleeve = pd.DataFrame({"equity": [25000, 25250], "gross": [10000, 0],
                           "net": [100, 0]}, index=dates)
    result, curve = multi.portfolio([(summary, sleeve, None)]*2, dates)
    assert result["final_equity"] == 100500
    assert result["return"] == pytest.approx(.005)
    assert curve.gross.iloc[0] == 20000
    assert result["commission"] == 4
    cash, _ = multi.portfolio([], dates)
    assert cash["final_equity"] == 100000


def test_missing_held_sleeve_mark_is_not_forward_filled():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    sleeve = pd.DataFrame({"equity": [25000], "gross": [10000], "net": [0]}, index=dates[:1])
    with pytest.raises(ValueError, match="calendar"):
        multi.portfolio([({}, sleeve, None)], dates)
