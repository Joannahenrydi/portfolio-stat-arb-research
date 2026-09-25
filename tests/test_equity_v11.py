import numpy as np
import pandas as pd

from scripts.evaluate_equity_v11 import internal_gap_lengths


def test_internal_gap_rule_ignores_prelisting_and_terminal_missing_rows():
    index = pd.bdate_range("2020-01-01", periods=12)
    volume = pd.DataFrame(
        {
            "clean_ipo": [np.nan, np.nan, 1, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan],
            "internal_gap": [1, 1, 1, np.nan, np.nan, np.nan, 1, 1, 1, 1, 1, 1],
            "zero_gap": [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        },
        index=index,
    )
    result = internal_gap_lengths(volume)
    assert result["clean_ipo"] == 0
    assert result["internal_gap"] == 3
    assert result["zero_gap"] == 2
