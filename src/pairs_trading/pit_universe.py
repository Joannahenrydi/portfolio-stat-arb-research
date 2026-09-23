"""Point-in-time eligibility with explicit provenance and an explicit session calendar.

This module validates *declared* evidence; it cannot authenticate a vendor archive or
create historical constituent/security-master data that the caller does not have.
A source publication reference or a contemporaneous capture is required for every
state/bar version. Download timestamps may establish availability from download
onward; they must never be backdated to the bar date.

Tables (all timestamps must carry a timezone):
* master: security_id, event_id, effective_at, available_at, publication_id,
  ticker, security_type, listing_country, exchange, status. Each event is a full
  state, not a patch. Revisions reuse event_id and its permanent security_id and
  effective_at. Symbols are labels, never join keys.
* bars: security_id, session, effective_at, available_at, publication_id,
  raw_close, volume, feed, is_stale, is_suspended. effective_at equals the supplied
  session close. Each (security_id, session, available_at) is one bar version.
* calendar: session (timezone-free ISO date), close_at (aware timestamp). Supply
  actual exchange sessions, including holidays/early closes and a close after
  decision_at, not a business-day
  range inferred from observed bars.
* publications: publication_id, published_at, evidence_ref, evidence_type,
  verified. evidence_type is source_publication or contemporaneous_capture;
  published_at is the independently evidenced publication/capture time. A capture
  of old bars today makes those bars available today, not on their historical dates.

Historical mode requires terminated instruments in the source's coverage. A
prospective archive may start with today's current snapshot, but cannot be used
before valid_from or represented as a historical universe. ETF holdings can be a
prospective candidate pool; they are not historical index membership evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class PITDataError(ValueError):
    """Input or provenance cannot support the requested point-in-time decision."""


@dataclass(frozen=True)
class PITManifest:
    source: str
    scope: str  # historical_verified | prospective_only
    master_kind: str  # historical_event_history | archived_snapshots | current_snapshot
    security_id_kind: str
    includes_terminated: bool
    availability_basis: str
    calendar_source: str
    valid_from: str | pd.Timestamp | None = None


@dataclass(frozen=True)
class UniversePolicy:
    target_size: int = 400
    min_members: int = 200
    min_raw_close: float = 5.0
    min_median_dollar_volume: float = 20_000_000.0
    liquidity_sessions: int = 60
    min_observations: int = 252
    min_coverage: float = 0.95
    stale_sessions: int = 5

    def __post_init__(self):
        for name in (
            "target_size",
            "min_members",
            "liquidity_sessions",
            "min_observations",
            "stale_sessions",
        ):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        if not np.isfinite(
            [self.min_raw_close, self.min_median_dollar_volume, self.min_coverage]
        ).all():
            raise ValueError("universe thresholds must be finite")
        if not 200 <= self.min_members <= self.target_size <= 500:
            raise ValueError("universe sizes must satisfy 200 <= min_members <= target_size <= 500")
        if self.min_raw_close < 5 or self.min_median_dollar_volume < 20_000_000:
            raise ValueError("raw-close and SIP dollar-volume floors cannot be relaxed")
        if self.liquidity_sessions != 60:
            raise ValueError("liquidity/coverage window must be 60 market sessions")
        if self.min_observations < 252 or not 0.95 <= self.min_coverage <= 1:
            raise ValueError("require at least 252 observations and 95% coverage")
        if self.stale_sessions < 1:
            raise ValueError("stale_sessions must be positive")


@dataclass(frozen=True)
class PITInputs:
    master: pd.DataFrame
    bars: pd.DataFrame
    calendar: pd.DataFrame
    publications: pd.DataFrame
    manifest: PITManifest


@dataclass(frozen=True)
class UniverseSnapshot:
    decision_at: pd.Timestamp
    latest_session: pd.Timestamp | None
    members: pd.DataFrame
    audit: pd.DataFrame
    data_gate_reasons: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def tradable(self) -> bool:
        return not self.data_gate_reasons

    @property
    def status(self) -> str:
        return "READY" if self.tradable else "DATA_GATE"

    def require_tradable(self) -> pd.DataFrame:
        if not self.tradable:
            raise PITDataError("DATA_GATE: " + "; ".join(self.data_gate_reasons))
        return self.members.copy()


_MASTER_COLUMNS = {
    "security_id",
    "event_id",
    "effective_at",
    "available_at",
    "publication_id",
    "ticker",
    "security_type",
    "listing_country",
    "exchange",
    "status",
}
_BAR_COLUMNS = {
    "security_id",
    "session",
    "effective_at",
    "available_at",
    "publication_id",
    "raw_close",
    "volume",
    "feed",
    "is_stale",
    "is_suspended",
}
_PUBLICATION_COLUMNS = {
    "publication_id",
    "published_at",
    "evidence_ref",
    "evidence_type",
    "verified",
}
_TERMINAL = {"DELISTED", "MERGED", "LIQUIDATED"}
_STATUSES = {"ACTIVE", "SUSPENDED"} | _TERMINAL
_US_EXCHANGES = {"XNYS", "XNAS", "XASE", "ARCX", "BATS", "IEXG"}
_AUDIT_COLUMNS = [
    "security_id",
    "ticker",
    "status",
    "latest_raw_close",
    "median_dollar_volume",
    "coverage",
    "valid_observations",
    "is_stale",
    "eligible",
    "exclusion_reasons",
    "selected",
]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise PITDataError(f"{name} must be a pandas DataFrame")
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise PITDataError(f"{name} missing columns: {missing}")
    return frame.copy()


def _time(value, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"{name} must be a timezone-aware timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise PITDataError(f"{name} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _times(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        # Reject naive values instead of silently interpreting them as UTC.
        frame[column] = pd.to_datetime(
            [_time(value, f"{name}.{column}") for value in frame[column]], utc=True
        )


def _dates(values, name: str) -> pd.DatetimeIndex:
    dates = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise PITDataError(f"{name} must contain timezone-free session dates")
        dates.append(stamp)
    return pd.DatetimeIndex(dates)


def _nonempty(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise PITDataError(f"{name}.{column} must contain nonempty strings")


def _bools(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        if not frame[column].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise PITDataError(f"{name}.{column} must contain explicit booleans")
        frame[column] = frame[column].astype(bool)


def _manifest(manifest: PITManifest, decision: pd.Timestamp) -> tuple[str, ...]:
    if not isinstance(manifest, PITManifest):
        raise PITDataError("manifest must be PITManifest")
    if not manifest.source.strip() or not manifest.calendar_source.strip():
        raise PITDataError("source and calendar_source evidence descriptions are required")
    if manifest.security_id_kind != "permanent":
        raise PITDataError("security_id must be a permanent security identifier, never a ticker")
    if manifest.availability_basis != "evidenced_publication_or_capture":
        raise PITDataError(
            "retrospective availability without publication/capture evidence is forbidden"
        )
    if manifest.scope == "historical_verified":
        if manifest.master_kind not in {"historical_event_history", "archived_snapshots"}:
            raise PITDataError("current_snapshot cannot establish a historical security master")
        if manifest.includes_terminated is not True:
            raise PITDataError("historical master must include terminated securities")
        return ()
    if manifest.scope != "prospective_only":
        raise PITDataError("scope must be historical_verified or prospective_only")
    if manifest.master_kind not in {
        "current_snapshot",
        "archived_snapshots",
        "historical_event_history",
    }:
        raise PITDataError("unsupported prospective master_kind")
    if manifest.valid_from is None:
        raise PITDataError("prospective_only requires valid_from")
    valid_from = _time(manifest.valid_from, "manifest.valid_from")
    if decision < valid_from:
        raise PITDataError("prospective_only data cannot support decisions before valid_from")
    return (
        "Prospective candidate pool only; no historical universe/index-membership claim.",
        "Historical feature bars first captured now are usable only from their evidenced availability.",
    )


def _evidence(frame: pd.DataFrame, publications: pd.DataFrame, name: str) -> None:
    if frame.empty:
        return
    linked = frame.merge(
        publications, on="publication_id", how="left", validate="many_to_one", indicator=True
    )
    if (linked["_merge"] != "both").any():
        raise PITDataError(f"{name} has missing publication evidence")
    if not linked["verified"].all():
        raise PITDataError(f"{name} has unverified publication evidence")
    if (linked["available_at"] < linked["published_at"]).any():
        raise PITDataError(f"{name} availability precedes evidenced publication/capture")


def _visible_states(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if master.empty:
        return master, master
    if master.duplicated(["event_id", "available_at"]).any():
        raise PITDataError("ambiguous master revisions at identical availability")
    identity = master.groupby("event_id")[["security_id", "effective_at"]].nunique()
    if (identity > 1).any().any():
        raise PITDataError("event_id must preserve immutable security_id and effective_at")
    events = (
        master.sort_values("available_at")
        .drop_duplicates("event_id", keep="last")
        .sort_values(["security_id", "effective_at"])
    )
    if events.duplicated(["security_id", "effective_at"]).any():
        raise PITDataError("ambiguous security state events at identical effective_at")
    if not events["status"].isin(_STATUSES).all():
        raise PITDataError("unknown security status")
    for _, history in events.groupby("security_id", sort=False):
        terminal = history["status"].isin(_TERMINAL).cummax()
        if (terminal & ~history["status"].isin(_TERMINAL)).any():
            raise PITDataError("terminal security_id cannot be reactivated or reused")
    state = events.groupby("security_id", sort=False).tail(1).copy()
    live = state[state["status"].isin({"ACTIVE", "SUSPENDED"})]
    if live.duplicated("ticker").any():
        raise PITDataError("ambiguous live ticker maps to multiple permanent security_id values")
    return state, events


def build_universe(
    inputs: PITInputs,
    decision_at: str | pd.Timestamp,
    policy: UniversePolicy | None = None,
) -> UniverseSnapshot:
    """Select a reproducible SIP-only universe at a timezone-aware decision time.

    No next-session or not-yet-published row is used, and no missing price is filled.
    ``members`` is empty on a data gate, even if some names pass. ``audit`` retains
    eligibility reasons and terminated securities visible at that decision.
    """
    policy = policy or UniversePolicy()
    decision = _time(decision_at, "decision_at")
    limitations = _manifest(inputs.manifest, decision)
    master = _require_columns(inputs.master, _MASTER_COLUMNS, "master")
    bars = _require_columns(inputs.bars, _BAR_COLUMNS, "bars")
    calendar = _require_columns(inputs.calendar, {"session", "close_at"}, "calendar")
    publications = _require_columns(inputs.publications, _PUBLICATION_COLUMNS, "publications")
    _times(master, ("effective_at", "available_at"), "master")
    _times(bars, ("effective_at", "available_at"), "bars")
    _times(calendar, ("close_at",), "calendar")
    _times(publications, ("published_at",), "publications")
    _nonempty(master, tuple(_MASTER_COLUMNS - {"effective_at", "available_at"}), "master")
    _nonempty(bars, ("security_id", "publication_id", "feed"), "bars")
    _nonempty(publications, ("publication_id", "evidence_ref", "evidence_type"), "publications")
    _bools(bars, ("is_stale", "is_suspended"), "bars")
    _bools(publications, ("verified",), "publications")
    if publications.duplicated("publication_id").any():
        raise PITDataError("publication_id must identify one immutable publication")
    if (
        not publications["evidence_type"]
        .isin({"source_publication", "contemporaneous_capture"})
        .all()
    ):
        raise PITDataError("unsupported or retrospective unverified publication evidence_type")
    calendar["session"] = _dates(calendar["session"], "calendar.session")
    calendar = calendar.sort_values("session")
    if calendar.empty or calendar["session"].duplicated().any():
        raise PITDataError("calendar must contain unique explicit market sessions")
    if not calendar["close_at"].is_monotonic_increasing or calendar["close_at"].duplicated().any():
        raise PITDataError("calendar closes must be strictly increasing")
    bars["session"] = _dates(bars["session"], "bars.session")

    # Hide future versions before any state/market-content checks. Schema must
    # still be valid, but future suspension or a revision never changes past state.
    master = master[(master["available_at"] <= decision) & (master["effective_at"] <= decision)]
    bars = bars[(bars["available_at"] <= decision) & (bars["effective_at"] <= decision)]
    _evidence(master, publications, "master")
    _evidence(bars, publications, "bars")
    if inputs.manifest.scope == "prospective_only":
        valid_from = _time(inputs.manifest.valid_from, "manifest.valid_from")
        if (master["available_at"] < valid_from).any():
            raise PITDataError("prospective master availability cannot predate valid_from")
        if (
            inputs.manifest.master_kind == "current_snapshot"
            and (master["effective_at"] < valid_from).any()
        ):
            raise PITDataError("current snapshot effective_at cannot be backdated")
    state, events = _visible_states(master)
    if not bars["feed"].eq("SIP").all():
        raise PITDataError("SIP consolidated bars are required; IEX cannot establish liquidity")
    if bars.duplicated(["security_id", "session", "available_at"]).any():
        raise PITDataError("ambiguous bar revisions at identical availability")
    known_ids = set(master["security_id"])
    if not set(bars["security_id"]).issubset(known_ids):
        raise PITDataError("visible bars reference security_id absent from visible master")
    joined = bars.merge(calendar, on="session", how="left", validate="many_to_one")
    if joined["close_at"].isna().any():
        raise PITDataError("visible bar session is absent from explicit market calendar")
    if not joined["effective_at"].eq(joined["close_at"]).all():
        raise PITDataError("bar effective_at must equal its market-session close")
    if (joined["available_at"] < joined["effective_at"]).any():
        raise PITDataError("final daily bar availability cannot precede session close")
    bars = bars.sort_values("available_at").drop_duplicates(["security_id", "session"], keep="last")
    sessions = calendar[calendar["close_at"] <= decision]
    latest = sessions["session"].iloc[-1] if not sessions.empty else None
    if not (calendar["close_at"] > decision).any():
        audit = pd.DataFrame(columns=_AUDIT_COLUMNS)
        return UniverseSnapshot(
            decision,
            latest,
            audit.copy(),
            audit,
            ("Calendar must extend beyond decision_at to establish the latest completed session.",),
            limitations,
        )
    if len(sessions) < max(policy.min_observations, policy.liquidity_sessions):
        audit = pd.DataFrame(columns=_AUDIT_COLUMNS)
        return UniverseSnapshot(
            decision,
            latest,
            audit.copy(),
            audit,
            ("Insufficient completed sessions in the supplied calendar.",),
            limitations,
        )
    session_index = pd.DatetimeIndex(sessions["session"])
    session_closes = sessions["close_at"].astype("int64").to_numpy()
    by_security = {sid: frame.set_index("session") for sid, frame in bars.groupby("security_id")}
    histories = {sid: frame for sid, frame in events.groupby("security_id")}
    rows = []
    for security in state.itertuples(index=False):
        frame = by_security.get(security.security_id, bars.iloc[:0].set_index("session"))
        frame = frame.reindex(session_index)  # Gaps count; there is deliberately no ffill.
        try:
            close = pd.to_numeric(frame["raw_close"], errors="raise").astype(float)
            volume = pd.to_numeric(frame["volume"], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise PITDataError("bar raw_close/volume must be numeric or missing") from exc
        valid = pd.Series(
            np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0),
            index=session_index,
        )
        suspended = frame["is_suspended"].eq(True)
        reported_stale = frame["is_stale"].eq(True)
        unchanged = (
            close.diff()
            .eq(0)
            .rolling(policy.stale_sessions, min_periods=policy.stale_sessions)
            .sum()
            .eq(policy.stale_sessions)
        )
        stale = reported_stale | unchanged
        valid &= ~suspended & ~stale
        if inputs.manifest.scope == "historical_verified":
            history = histories[security.security_id]
            effective = history["effective_at"].astype("int64").to_numpy()
            positions = np.searchsorted(effective, session_closes, side="right") - 1
            historical_status = history["status"].to_numpy()[np.maximum(positions, 0)]
            valid &= (positions >= 0) & (historical_status == "ACTIVE")
        recent = valid.iloc[-policy.liquidity_sessions :]
        coverage = float(recent.mean())
        dollar_volume = (close * volume).where(valid).iloc[-policy.liquidity_sessions :]
        median = float(dollar_volume.median()) if dollar_volume.notna().any() else np.nan
        observations = int(valid.sum())
        reasons = []
        if security.status != "ACTIVE":
            reasons.append("inactive_or_suspended")
        if security.security_type != "COMMON_STOCK":
            reasons.append("not_common_stock")
        if security.listing_country != "US" or security.exchange not in _US_EXCHANGES:
            reasons.append("not_us_listed_stock")
        if not bool(valid.iloc[-1]):
            reasons.append("missing_stale_or_invalid_latest_bar")
        if bool(stale.iloc[-1]):
            reasons.append("stale")
        if not np.isfinite(close.iloc[-1]) or close.iloc[-1] < policy.min_raw_close:
            reasons.append("raw_close_below_floor")
        if not np.isfinite(median) or median < policy.min_median_dollar_volume:
            reasons.append("sip_liquidity_below_floor")
        if observations < policy.min_observations:
            reasons.append("insufficient_valid_observations")
        if coverage < policy.min_coverage:
            reasons.append("session_coverage_below_floor")
        rows.append(
            {
                "security_id": security.security_id,
                "ticker": security.ticker,
                "status": security.status,
                "latest_raw_close": float(close.iloc[-1]),
                "median_dollar_volume": median,
                "coverage": coverage,
                "valid_observations": observations,
                "is_stale": bool(stale.iloc[-1]),
                "eligible": not reasons,
                "exclusion_reasons": tuple(reasons),
                "selected": False,
            }
        )
    audit = pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
    candidates = (
        audit[audit["eligible"].eq(True)]
        .sort_values(
            ["median_dollar_volume", "security_id"], ascending=[False, True], kind="stable"
        )
        .head(policy.target_size)
    )
    gates = ()
    if len(candidates) < policy.min_members:
        gates = (
            (
                f"Only {len(candidates)} eligible stocks; minimum {policy.min_members} required. "
                "No tradable universe produced."
            ),
        )
        members = audit.iloc[:0].copy()
    else:
        audit.loc[candidates.index, "selected"] = True
        members = audit.loc[candidates.index].reset_index(drop=True)
    return UniverseSnapshot(
        decision, latest, members, audit.reset_index(drop=True), gates, limitations
    )
