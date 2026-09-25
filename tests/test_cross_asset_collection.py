from scripts.collect_cross_asset_etfs import UNIVERSE


def test_cross_asset_universe_has_all_frozen_risk_sleeves():
    assert len(UNIVERSE) == len(set(UNIVERSE)) == 19
    for symbol in ("SPY", "TLT", "HYG", "GLD", "DBC", "UUP"):
        assert symbol in UNIVERSE
