"""Frozen expanded ETF universe and coarse ex-ante macro risk metadata for v17."""

from __future__ import annotations

import pandas as pd

UNIVERSE = [
    "SPY", "QQQ", "IWM", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI",
    "XLB", "XLK", "XLU", "EFA", "EEM", "EWJ", "EWU", "EWZ", "FXI", "INDA",
    "VNQ", "BIL", "SHY", "IEF", "TLT", "TIP", "GOVT", "LQD", "HYG", "EMB",
    "MUB", "GLD", "SLV", "DBC", "PDBC", "USO", "UNG", "DBA", "CPER", "CORN",
    "WEAT", "UUP", "FXE", "FXY", "FXB",
]


def factor_loadings() -> pd.DataFrame:
    frame = pd.DataFrame(
        0.0, index=UNIVERSE, columns=["equity", "duration", "credit", "commodity", "usd"]
    )
    frame.loc[["SPY", "QQQ", "IWM"], "equity"] = [1.0, 1.1, 1.1]
    sectors = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU"]
    frame.loc[sectors, "equity"] = 1.0
    frame.loc[["EFA", "EEM", "EWJ", "EWU", "EWZ", "FXI", "INDA"], "equity"] = [
        .9, 1.0, .8, .8, 1.1, 1.0, 1.0,
    ]
    frame.loc[["VNQ", "LQD", "HYG", "EMB"], "equity"] = [.7, .1, .4, .2]
    frame.loc[["BIL", "SHY", "IEF", "TLT", "TIP", "GOVT", "LQD", "HYG", "EMB", "MUB"],
              "duration"] = [.1, 1, 4, 15, 7, 6, 8, 3, 7, 6]
    frame.loc[["LQD", "HYG", "EMB", "MUB"], "credit"] = [.4, 1.0, .8, .2]
    frame.loc[["GLD", "SLV", "DBC", "PDBC", "USO", "UNG", "DBA", "CPER", "CORN", "WEAT"],
              "commodity"] = [.3, .5, 1, 1, 1.2, 1.5, .8, 1, 1, 1]
    frame.loc["UUP", "usd"] = 1.0
    frame.loc[["FXE", "FXY", "FXB"], "usd"] = -1.0
    frame.loc[["EFA", "EEM", "EWJ", "EWU", "EWZ", "FXI", "INDA"], "usd"] = -.3
    frame.loc[["GLD", "SLV", "DBC", "PDBC", "USO", "UNG", "DBA", "CPER", "CORN", "WEAT"],
              "usd"] = -.3
    return frame


def asset_sleeves() -> pd.Series:
    mapping = {}
    for symbol in UNIVERSE:
        if symbol in {"SPY", "QQQ", "IWM", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV",
                      "XLI", "XLB", "XLK", "XLU", "EFA", "EEM", "EWJ", "EWU", "EWZ",
                      "FXI", "INDA", "VNQ"}:
            mapping[symbol] = "equity"
        elif symbol in {"BIL", "SHY", "IEF", "TLT", "TIP", "GOVT"}:
            mapping[symbol] = "rates"
        elif symbol in {"LQD", "HYG", "EMB", "MUB"}:
            mapping[symbol] = "credit"
        elif symbol in {"GLD", "SLV"}:
            mapping[symbol] = "metals"
        elif symbol in {"DBC", "PDBC", "USO", "UNG", "DBA", "CPER", "CORN", "WEAT"}:
            mapping[symbol] = "commodity"
        else:
            mapping[symbol] = "usd"
    return pd.Series(mapping, name="sleeve").reindex(UNIVERSE)
