import numpy as np
import pandas as pd

from scripts.evaluate_equity_v10 import tail_select


def test_tail_selection_is_cross_sectional_and_preserves_only_frozen_tails():
    columns = [f"S{i:03d}" for i in range(100)]
    index = pd.bdate_range("2024-01-02", periods=2)
    score = pd.DataFrame(
        [np.arange(100), np.arange(100)[::-1]], index=index, columns=columns, dtype=float
    )
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    selected = tail_select(score, eligibility, 0.15)
    assert selected.notna().sum(axis=1).eq(30).all()
    assert selected.loc[index[0], "S014"] == 14
    assert np.isnan(selected.loc[index[0], "S015"])
    assert np.isnan(selected.loc[index[0], "S084"])
    assert selected.loc[index[0], "S085"] == 85


def test_tail_selection_never_revives_ineligible_names():
    index = pd.bdate_range("2024-01-02", periods=1)
    columns = [f"S{i:03d}" for i in range(100)]
    score = pd.DataFrame([np.arange(100)], index=index, columns=columns, dtype=float)
    eligibility = pd.DataFrame(True, index=index, columns=columns)
    eligibility.loc[index[0], ["S000", "S099"]] = False
    selected = tail_select(score, eligibility, 0.15)
    assert selected.loc[index[0], ["S000", "S099"]].isna().all()
