import numpy as np
import pandas as pd

from scripts.evaluate_cross_asset_v12 import pair_score


def test_pair_score_maps_equal_and_opposite_legs():
    index = pd.bdate_range("2020-01-01", periods=3)
    signal = pd.Series([1.0, -2.0, np.nan], index=index)
    result = pair_score({("A", "B"): signal}, index, ["A", "B", "C"])
    assert result.loc[index[0], "A"] == 1
    assert result.loc[index[0], "B"] == -1
    assert result.loc[index[1], "A"] == -2
    assert result.loc[index[1], "B"] == 2
    assert result["C"].isna().all()
