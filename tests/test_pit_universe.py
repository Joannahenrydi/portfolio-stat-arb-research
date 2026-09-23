from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pairs_trading.pit_universe import (
    PITDataError,
    PITInputs,
    PITManifest,
    UniversePolicy,
    build_universe,
)


def fixture(names=1, periods=270):
    # Synthetic explicit market calendar. The omitted weekday verifies that
    # eligibility uses this table, not an inferred Monday-Friday calendar.
    dates = pd.bdate_range("2022-01-03", periods=periods + 8).delete(12)
    closes = dates.tz_localize("UTC") + pd.Timedelta(hours=21)
    calendar = pd.DataFrame({"session": dates, "close_at": closes})
    master_time = closes[0] - pd.Timedelta(days=1)
    master = pd.DataFrame(
        [
            {
                "security_id": f"permanent-{n:04d}",
                "event_id": f"listing-{n:04d}",
                "effective_at": master_time,
                "available_at": master_time,
                "publication_id": "master",
                "ticker": f"S{n:04d}",
                "security_type": "COMMON_STOCK",
                "listing_country": "US",
                "exchange": "XNAS",
                "status": "ACTIVE",
            }
            for n in range(names)
        ]
    )
    frames = []
    for n in range(names):
        frames.append(
            pd.DataFrame(
                {
                    "security_id": f"permanent-{n:04d}",
                    "session": dates[:periods],
                    "effective_at": closes[:periods],
                    "available_at": closes[:periods] + pd.Timedelta(minutes=15),
                    "publication_id": [f"bar-{k}" for k in range(periods)],
                    "raw_close": 10 + np.arange(periods) / 1000,
                    "volume": 3_000_000.0,
                    "feed": "SIP",
                    "is_stale": False,
                    "is_suspended": False,
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True)
    publications = pd.DataFrame(
        {
            "publication_id": ["master"] + [f"bar-{k}" for k in range(periods)],
            "published_at": [master_time] + list(closes[:periods] + pd.Timedelta(minutes=15)),
            "evidence_ref": "fixture://contemporaneous-archive",
            "evidence_type": "source_publication",
            "verified": True,
        }
    )
    manifest = PITManifest(
        source="Synthetic test archive, never a real-data claim",
        scope="historical_verified",
        master_kind="historical_event_history",
        security_id_kind="permanent",
        includes_terminated=True,
        availability_basis="evidenced_publication_or_capture",
        calendar_source="Synthetic explicit test session fixture",
    )
    return PITInputs(master, bars, calendar, publications, manifest), closes[
        periods - 1
    ] + pd.Timedelta(hours=1)


def publication(inputs, identifier, at):
    return pd.concat(
        [
            inputs.publications,
            pd.DataFrame(
                [
                    {
                        "publication_id": identifier,
                        "published_at": at,
                        "evidence_ref": f"fixture://{identifier}",
                        "evidence_type": "source_publication",
                        "verified": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


def audit_row(result, sid="permanent-0000"):
    return result.audit.set_index("security_id").loc[sid]


def test_default_selects_400_from_405_with_deterministic_permanent_id_tie_break():
    inputs, decision = fixture(names=405)
    result = build_universe(inputs, decision)
    assert result.tradable and result.status == "READY"
    assert len(result.require_tradable()) == 400
    assert result.members.security_id.tolist() == [f"permanent-{n:04d}" for n in range(400)]
    assert result.audit.selected.sum() == 400
    assert result.audit.eligible.sum() == 405
    assert result.members.coverage.eq(1).all()
    assert result.members.valid_observations.eq(270).all()


def test_below_minimum_is_data_gate_with_no_tradeable_members():
    inputs, decision = fixture(names=2)
    result = build_universe(inputs, decision)
    assert not result.tradable and result.status == "DATA_GATE"
    assert result.members.empty
    assert result.audit.eligible.all()
    assert not result.audit.selected.any()
    with pytest.raises(PITDataError, match="minimum 200"):
        result.require_tradable()


def test_late_master_and_bar_revisions_do_not_change_historical_decision():
    inputs, decision = fixture()
    before = build_universe(inputs, decision)
    later = decision + pd.Timedelta(days=2)
    revision = inputs.bars.iloc[[-1]].copy()
    revision["available_at"] = later
    revision["publication_id"] = "late"
    revision["raw_close"] = 1.0
    state = inputs.master.copy()
    state["status"] = "SUSPENDED"
    state["available_at"] = later
    state["publication_id"] = "late"
    revised = replace(
        inputs,
        bars=pd.concat([inputs.bars, revision]),
        master=pd.concat([inputs.master, state]),
        publications=publication(inputs, "late", later),
    )
    pd.testing.assert_frame_equal(before.audit, build_universe(revised, decision).audit)
    assert audit_row(build_universe(revised, later)).status == "SUSPENDED"


def test_future_effective_delisting_retains_historical_membership():
    inputs, decision = fixture()
    terminal = inputs.master.copy()
    terminal["event_id"] = "delisting-0000"
    terminal["effective_at"] = decision + pd.Timedelta(days=1)
    terminal["available_at"] = decision - pd.Timedelta(hours=1)
    terminal["publication_id"] = "announcement"
    terminal["status"] = "DELISTED"
    history = replace(
        inputs,
        master=pd.concat([inputs.master, terminal]),
        publications=publication(inputs, "announcement", decision - pd.Timedelta(hours=1)),
    )
    assert audit_row(build_universe(history, decision)).eligible
    after = audit_row(build_universe(history, decision + pd.Timedelta(days=1)))
    assert after.status == "DELISTED"
    assert not after.eligible
    assert "inactive_or_suspended" in after.exclusion_reasons


def test_coverage_counts_missing_market_sessions_and_does_not_forward_fill():
    inputs, decision = fixture()
    missing = inputs.bars.session.iloc[-5:-1]
    gaps = replace(inputs, bars=inputs.bars[~inputs.bars.session.isin(missing)])
    row = audit_row(build_universe(gaps, decision))
    assert row.valid_observations == 266
    assert row.coverage == pytest.approx(56 / 60)
    assert "session_coverage_below_floor" in row.exclusion_reasons
    # Exactly three missing sessions still satisfies the stipulated 95% floor.
    gaps = replace(inputs, bars=inputs.bars.drop(inputs.bars.index[-4:-1]))
    row = audit_row(build_universe(gaps, decision))
    assert row.coverage == pytest.approx(0.95)
    assert row.eligible
    # A missing latest session remains ineligible even with 98.3% coverage.
    last_gap = replace(inputs, bars=inputs.bars.iloc[:-1])
    row = audit_row(build_universe(last_gap, decision))
    assert row.coverage == pytest.approx(59 / 60)
    assert "missing_stale_or_invalid_latest_bar" in row.exclusion_reasons


def test_252_observations_means_valid_bars_not_calendar_age():
    inputs, decision = fixture()
    sparse = replace(inputs, bars=inputs.bars.iloc[19:])
    row = audit_row(build_universe(sparse, decision))
    assert row.coverage == 1
    assert row.valid_observations == 251
    assert "insufficient_valid_observations" in row.exclusion_reasons


def test_raw_price_liquidity_stock_type_and_staleness_filters():
    inputs, decision = fixture(names=5)
    bars = inputs.bars.copy()
    bars.loc[bars.security_id.eq("permanent-0000"), "raw_close"] *= 0.1
    bars.loc[bars.security_id.eq("permanent-0001"), "volume"] = 1_000_000
    last_six = bars[bars.security_id.eq("permanent-0002")].tail(6).index
    bars.loc[last_six, "raw_close"] = 10.5
    bars.loc[bars.security_id.eq("permanent-0003"), "is_suspended"] = True
    master = inputs.master.copy()
    master.loc[master.security_id.eq("permanent-0004"), "security_type"] = "ETF"
    result = build_universe(replace(inputs, bars=bars, master=master), decision)
    assert "raw_close_below_floor" in audit_row(result).exclusion_reasons
    assert "sip_liquidity_below_floor" in audit_row(result, "permanent-0001").exclusion_reasons
    assert "stale" in audit_row(result, "permanent-0002").exclusion_reasons
    assert audit_row(result, "permanent-0003").valid_observations == 0
    assert "not_common_stock" in audit_row(result, "permanent-0004").exclusion_reasons


def test_sip_is_required_even_if_iex_volume_passes_threshold():
    inputs, decision = fixture()
    bars = inputs.bars.assign(feed="IEX", volume=1e10)
    with pytest.raises(PITDataError, match="SIP consolidated"):
        build_universe(replace(inputs, bars=bars), decision)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda m: replace(m, master_kind="current_snapshot"), "current_snapshot"),
        (lambda m: replace(m, includes_terminated=False), "terminated"),
        (lambda m: replace(m, security_id_kind="ticker"), "permanent"),
        (lambda m: replace(m, availability_basis="assume_next_day"), "retrospective availability"),
    ],
)
def test_unsubstantiated_historical_master_is_rejected(mutator, match):
    inputs, decision = fixture()
    with pytest.raises(PITDataError, match=match):
        build_universe(replace(inputs, manifest=mutator(inputs.manifest)), decision)


def test_missing_unverified_or_backdated_publication_evidence_is_rejected():
    inputs, decision = fixture()
    with pytest.raises(PITDataError, match="missing publication"):
        build_universe(replace(inputs, publications=inputs.publications.iloc[:-1]), decision)
    unverified = inputs.publications.copy()
    unverified.loc[0, "verified"] = False
    with pytest.raises(PITDataError, match="unverified publication"):
        build_universe(replace(inputs, publications=unverified), decision)
    backdated = inputs.publications.copy()
    backdated.loc[0, "published_at"] = decision
    with pytest.raises(PITDataError, match="precedes evidenced"):
        build_universe(replace(inputs, publications=backdated), decision)


def test_current_capture_is_prospectively_valid_but_cannot_backtest_past_decisions():
    inputs, decision = fixture()
    capture = decision
    manifest = replace(
        inputs.manifest,
        scope="prospective_only",
        master_kind="current_snapshot",
        includes_terminated=False,
        valid_from=capture,
    )
    master = inputs.master.assign(
        effective_at=capture, available_at=capture, publication_id="capture"
    )
    bars = inputs.bars.assign(available_at=capture, publication_id="capture")
    pubs = pd.DataFrame(
        [
            {
                "publication_id": "capture",
                "published_at": capture,
                "evidence_ref": "fixture://captured-current-assets-and-old-bars",
                "evidence_type": "contemporaneous_capture",
                "verified": True,
            }
        ]
    )
    prospective = replace(inputs, master=master, bars=bars, publications=pubs, manifest=manifest)
    result = build_universe(prospective, decision)
    assert audit_row(result).eligible
    assert result.limitations
    with pytest.raises(PITDataError, match="before valid_from"):
        build_universe(prospective, decision - pd.Timedelta(seconds=1))
    with pytest.raises(PITDataError, match="cannot be backdated"):
        build_universe(
            replace(prospective, master=master.assign(effective_at=inputs.master.effective_at)),
            decision,
        )


def test_terminal_permanent_id_cannot_be_reused_and_ticker_cannot_be_ambiguous():
    inputs, decision = fixture(names=2)
    terminal = inputs.master.iloc[[0]].copy()
    terminal["event_id"] = "terminal-0"
    terminal["effective_at"] += pd.Timedelta(days=10)
    terminal["status"] = "DELISTED"
    resurrected = terminal.copy()
    resurrected["event_id"] = "new-company-illegal-id-reuse"
    resurrected["effective_at"] += pd.Timedelta(days=10)
    resurrected["status"] = "ACTIVE"
    resurrected["ticker"] = "NEWCO"
    with pytest.raises(PITDataError, match="terminal security_id"):
        build_universe(
            replace(inputs, master=pd.concat([inputs.master, terminal, resurrected])), decision
        )
    ambiguous = inputs.master.assign(ticker="SAME")
    with pytest.raises(PITDataError, match="ambiguous live ticker"):
        build_universe(replace(inputs, master=ambiguous), decision)


def test_event_revision_cannot_change_permanent_identity():
    inputs, decision = fixture()
    revision = inputs.master.copy()
    revision["security_id"] = "different-permanent-id"
    revision["available_at"] += pd.Timedelta(hours=1)
    with pytest.raises(PITDataError, match="immutable security_id"):
        build_universe(replace(inputs, master=pd.concat([inputs.master, revision])), decision)


def test_ticker_change_preserves_identity_and_bar_history():
    inputs, decision = fixture()
    renamed = inputs.master.copy()
    renamed["event_id"] = "rename-0"
    renamed["ticker"] = "RENAMED"
    renamed["effective_at"] = decision - pd.Timedelta(days=30)
    result = build_universe(replace(inputs, master=pd.concat([inputs.master, renamed])), decision)
    assert audit_row(result).ticker == "RENAMED"
    assert audit_row(result).valid_observations == 270
    assert audit_row(result).eligible


def test_supplied_calendar_rejects_non_session_bar_and_final_bar_before_close():
    inputs, decision = fixture()
    with pytest.raises(PITDataError, match="absent from explicit market calendar"):
        build_universe(replace(inputs, calendar=inputs.calendar.iloc[1:]), decision)
    wrong_effective = inputs.bars.copy()
    wrong_effective.loc[0, "effective_at"] -= pd.Timedelta(minutes=1)
    with pytest.raises(PITDataError, match="must equal"):
        build_universe(replace(inputs, bars=wrong_effective), decision)
    with pytest.raises(PITDataError, match="timezone-aware"):
        build_universe(inputs, "2023-01-01")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_size": 199},
        {"target_size": 501},
        {"min_members": 1},
        {"min_raw_close": 4},
        {"min_median_dollar_volume": 1e6},
        {"liquidity_sessions": 20},
        {"min_observations": 100},
        {"min_coverage": 0.9},
    ],
)
def test_policy_cannot_weaken_economic_requirements(kwargs):
    with pytest.raises((ValueError, TypeError)):
        UniversePolicy(**kwargs)


def test_calendar_must_extend_past_decision_to_prove_latest_session_freshness():
    inputs, decision = fixture()
    truncated = inputs.calendar[inputs.calendar.close_at <= decision]
    result = build_universe(replace(inputs, calendar=truncated), decision)
    assert result.status == "DATA_GATE"
    assert result.members.empty
    assert "extend beyond decision_at" in result.data_gate_reasons[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_raw_close": float("nan")},
        {"min_median_dollar_volume": float("inf")},
        {"target_size": 400.5},
        {"min_observations": 252.1},
    ],
)
def test_policy_rejects_nonfinite_or_noninteger_thresholds(kwargs):
    with pytest.raises((ValueError, TypeError)):
        UniversePolicy(**kwargs)
