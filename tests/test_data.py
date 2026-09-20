import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pairs_trading.data import download_daily


class DataTests(unittest.TestCase):
    def test_aligns_without_forward_fill_and_records_provenance(self):
        def ticker(symbol):
            dates = ["2020-01-02", "2020-01-03"] if symbol == "X" else ["2020-01-03"]
            frame = pd.DataFrame({"Adj Close": [10.0] * len(dates)},
                                 index=pd.DatetimeIndex(dates, tz="America/New_York"))
            return types.SimpleNamespace(history=lambda **kwargs: frame)

        fake = types.SimpleNamespace(Ticker=ticker, __version__="test",
                                     set_tz_cache_location=lambda path: None)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"yfinance": fake}):
            prices = download_daily(["X", "Y"], "2020-01-01", "2020-02-01", directory)
            self.assertEqual(len(prices), 1)
            self.assertEqual(str(prices.index[0].date()), "2020-01-03")
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            self.assertEqual(metadata["dropped_noncommon_dates"], 1)
            self.assertEqual(len(metadata["sha256"]), 64)
