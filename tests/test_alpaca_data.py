import json

import pytest

from pairs_trading.alpaca_data import download_alpaca


def bar():
    return {"t": "2025-01-02T05:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}


def test_paginates_across_symbols_and_adjustments(tmp_path):
    calls = []

    def fetch(params):
        calls.append(params)
        if "page_token" not in params:
            return {"bars": {"EWA": [bar()]}, "next_page_token": "next"}
        return {"bars": {"EWC": [bar()]}, "next_page_token": None}

    result = download_alpaca(["EWA", "EWC"], "2025-01-01", "2025-01-03",
                             tmp_path / "data", fetch=fetch)
    assert len(calls) == 4
    assert [c["adjustment"] for c in calls] == ["raw", "raw", "all", "all"]
    assert all(c["feed"] == "iex" for c in calls)
    assert result["files"]["bars_raw.csv"]["rows_by_symbol"] == {"EWA": 1, "EWC": 1}
    assert result["status"] == "downloaded_not_execution_validated"


def test_duplicate_or_missing_data_rejected(tmp_path):
    for name, bars in [("duplicate", {"EWA": [bar(), bar()]}), ("missing", {})]:
        with pytest.raises(ValueError):
            download_alpaca(["EWA"], "2025-01-01", "2025-01-03", tmp_path / name,
                            fetch=lambda params, bars=bars: {"bars": bars})
        metadata = json.loads((tmp_path / name / "metadata.json").read_text())
        assert metadata["status"] == "incomplete"


def test_credentials_are_required_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="locally"):
        download_alpaca(["EWA"], "2025-01-01", "2025-01-03", tmp_path / "data")
