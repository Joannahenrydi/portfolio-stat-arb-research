"""Causal adaptive forecasts, constrained dollar hedges and cost-aware execution."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .research import Costs, Parameters, research_signals, total_return_prices


@dataclass(frozen=True)
class DynamicParameters:
    train_window: int = 504
    horizon: int = 5
    ridge: float = .1
    confidence: float = .1
    features: str = "full"
    covariance_span: int = 60
    minimum_correlation: float = .3
    target_account_volatility: float = .02
    rebalance_band: float = .04
    hedge_mode: str = "minimum_variance"


def optimal_x_weight(var_x, var_y, covariance, previous=.5):
    """Minimize gross-normalized pair variance plus balance/turnover penalties.

    Long Y and short X: return=(1-a)*rY-a*rX. Constrain a to [0.4,0.6].
    Penalties are lambda*(a-.5)^2 + eta*(a-previous)^2, scaled by trace.
    """
    trace = var_x + var_y
    shrink, inertia = .1 * trace, .5 * trace
    denominator = var_x + var_y + 2 * covariance + shrink + inertia
    if not np.isfinite(denominator) or denominator <= 1e-15:
        return .5
    return float(np.clip((var_y + covariance + shrink * .5 + inertia * previous)
                         / denominator, .4, .6))


def dynamic_signals(market, params):
    tr = total_return_prices(market)
    returns = tr.pct_change()
    vx = returns.X.ewm(span=params.covariance_span, min_periods=60).var()
    vy = returns.Y.ewm(span=params.covariance_span, min_periods=60).var()
    cov = returns.X.ewm(span=params.covariance_span, min_periods=60).cov(returns.Y)
    weights, previous = [], .5
    for xx, yy, xy in zip(vx, vy, cov):
        previous = optimal_x_weight(xx, yy, xy, previous)
        weights.append(previous)
    a = pd.Series(weights, index=market.index)
    if params.hedge_mode == "equal":
        a[:] = .5
    elif params.hedge_mode != "minimum_variance":
        raise ValueError("unknown hedge mode")
    pair_sigma = np.sqrt((a*a*vx + (1-a)**2*vy - 2*a*(1-a)*cov).clip(lower=1e-12))
    corr = cov / np.sqrt(vx*vy)
    log_ratio = np.log(tr.Y / tr.X)
    z = (log_ratio - log_ratio.rolling(60).mean()) / log_ratio.rolling(60).std()
    features = pd.DataFrame({"z": z, "change_z": z.diff(),
                             "relative_5": tr.Y.pct_change(5)-tr.X.pct_change(5),
                             "relative_20": tr.Y.pct_change(20)-tr.X.pct_change(20),
                             "relative_60": tr.Y.pct_change(60)-tr.X.pct_change(60)}, index=market.index)
    if params.features == "simple":
        features = features[["z", "relative_5"]]
    elif params.features == "kalman":
        kalman = research_signals(market, Parameters(window=120, process_variance=1e-7))
        features = pd.DataFrame({"kalman_z": kalman.z}, index=market.index)
        z = kalman.z
    # t signal -> t+1 fill -> t+1+h outcome; only matured labels may train.
    h = params.horizon
    future_x = tr.X.shift(-(h+1)) / tr.X.shift(-1) - 1
    future_y = tr.Y.shift(-(h+1)) / tr.Y.shift(-1) - 1
    labels = (1-a)*future_y-a*future_x
    predictions = np.full(len(market), np.nan)
    last_label_indices = np.full(len(market), -1, dtype=int)
    trained_samples = np.zeros(len(market), dtype=int)
    coefficients = []
    fitted = None
    feature_values, outcomes = features.to_numpy(), labels.to_numpy()
    last_month = None
    for i, date in enumerate(market.index):
        month = (date.year, date.month)
        if month != last_month:
            stop = i - h  # exclusive: last training label is i-h-1, matures at i
            begin = max(0, stop-params.train_window)
            if stop > begin:
                xx, yy = feature_values[begin:stop], outcomes[begin:stop]
                valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy)
                if valid.sum() >= 180:
                    xx, yy = xx[valid], yy[valid]
                    mean, scale = xx.mean(axis=0), xx.std(axis=0)
                    scale = np.maximum(scale, 1e-10)
                    xx = (xx-mean)/scale
                    intercept = yy.mean()
                    beta = np.linalg.solve(xx.T@xx + len(xx)*params.ridge*np.eye(xx.shape[1]),
                                           xx.T@(yy-intercept))
                    if params.features == "kalman":
                        # A symmetric reversion hypothesis: no directional intercept.
                        mean = np.zeros_like(mean)
                        xx = feature_values[begin:stop][valid]/scale
                        intercept = 0.
                        beta = np.linalg.solve(xx.T@xx + len(xx)*params.ridge*np.eye(xx.shape[1]), xx.T@yy)
                        beta = np.minimum(beta, 0.)
                    fitted = (mean, scale, beta, intercept, stop-1, len(xx))
                    coefficients.append({"date": date, "last_label_index": stop-1,
                                         "last_label_maturity": market.index[stop-1+h+1],
                                         "samples": len(xx), "intercept": float(intercept),
                                         **{name: float(value) for name, value in zip(features.columns, beta)}})
            last_month = month
        if fitted is not None and np.isfinite(feature_values[i]).all():
            mean, scale, beta, intercept, last_label, count = fitted
            predictions[i] = float(np.clip((feature_values[i]-mean)/scale, -5, 5)@beta+intercept)
            last_label_indices[i], trained_samples[i] = last_label, count
    result = pd.DataFrame({"prediction": predictions, "weight_x": a, "weight_y": 1-a,
                           "share_ratio_x_per_y": (a/(1-a))*(market.Y/market.X),
                           "sigma_daily": pair_sigma, "correlation": corr,
                           "volatility_ratio": pair_sigma/pair_sigma.rolling(126).median(),
                           "last_label_index": last_label_indices, "training_samples": trained_samples,
                           "z": z, "var_x": vx, "var_y": vy, "covariance": cov}, index=market.index)
    return result, pd.DataFrame(coefficients)


def desired_position(equity, px, py, signal, params, costs, cap):
    mu, a, sigma = signal["prediction"], signal["weight_x"], signal["sigma_daily"]
    if (cap <= 0 or not np.isfinite([mu, a, sigma, signal["correlation"], signal["volatility_ratio"]]).all()
            or sigma <= 0 or signal["correlation"] < params.minimum_correlation
            or signal["volatility_ratio"] > 2.5):
        return 0, 0, float("nan"), 0.
    fraction = min(cap, params.target_account_volatility/(sigma*np.sqrt(252)))
    gross = max(equity, 0)*fraction
    side = 1 if mu > 0 else -1
    qx, qy = -side*int(gross*a/px), side*int(gross*(1-a)/py)
    actual_gross = abs(qx)*px+abs(qy)*py
    if not qx or not qy:
        return 0, 0, float("nan"), 0.
    round_trip = sum(costs.execution((qx, qy), (px, py)))+sum(costs.execution((-qx, -qy), (px, py)))
    borrow = (max(-qx, 0)*px+max(-qy, 0)*py)*costs.borrow_annual*(params.horizon*1.5+1)/365
    threshold = (round_trip+borrow)/actual_gross+params.confidence*sigma*np.sqrt(params.horizon)
    if abs(mu) <= threshold:
        return 0, 0, threshold, fraction
    # Scale marginal signals down; execution minimums must be checked again.
    strength = float(np.clip((abs(mu)-threshold)/max(sigma*np.sqrt(params.horizon), 1e-8), .25, 1))
    qx, qy = -side*int(gross*strength*a/px), side*int(gross*strength*(1-a)/py)
    actual_gross = abs(qx)*px+abs(qy)*py
    if not qx or not qy:
        return 0, 0, threshold, fraction*strength
    round_trip = sum(costs.execution((qx, qy), (px, py)))+sum(costs.execution((-qx, -qy), (px, py)))
    borrow = (max(-qx, 0)*px+max(-qy, 0)*py)*costs.borrow_annual*(params.horizon*1.5+1)/365
    threshold = (round_trip+borrow)/actual_gross+params.confidence*sigma*np.sqrt(params.horizon)
    if abs(mu) <= threshold:
        qx = qy = 0
    return qx, qy, threshold, fraction*strength


def simulate_dynamic(market, signals, params, start, end, cap=.2, capital=100000, costs=None):
    costs = costs if costs is not None else Costs()
    if not 0 <= cap <= 1 or capital <= 0:
        raise ValueError("invalid capital or exposure cap")
    if any(v < 0 for v in vars(costs).values()):
        raise ValueError("costs must be nonnegative")
    data = market.join(signals).loc[start:end]
    if len(data) < 2:
        raise ValueError("insufficient evaluation data")
    equity, peak = float(capital), float(capital)
    qx = qy = held = since_rebalance = 0
    pending = None
    stopped = False
    commission_total = slippage_total = borrow_total = dividends_total = 0.
    records, trades = [], []
    previous = None
    for i, (date, row) in enumerate(data.iterrows()):
        px, py = row.X, row.Y
        if previous is not None:
            old_date, old_x, old_y = previous
            dividend = qx*row.div_X+qy*row.div_Y
            borrow = (max(-qx, 0)*old_x+max(-qy, 0)*old_y)*costs.borrow_annual*(date-old_date).days/365
            equity += qx*(px-old_x)+qy*(py-old_y)+dividend-borrow
            dividends_total += dividend
            borrow_total += borrow
        if pending is not None:
            event, nx, ny, signal_date = pending
            # Quantities are fixed at the previous close. Skip entries exceeding the cap after a gap.
            if event != "ENTRY" or abs(nx)*px+abs(ny)*py <= max(equity, 0)*cap:
                dx, dy = nx-qx, ny-qy
                commission, slip = costs.execution((dx, dy), (px, py))
                equity -= commission+slip
                commission_total += commission
                slippage_total += slip
                trades.append({"date": date, "signal_date": signal_date, "event": event,
                               "delta_qx": dx, "delta_qy": dy, "qx": nx, "qy": ny,
                               "px": px, "py": py, "commission": commission, "slippage": slip})
                if event in ("ENTRY", "EXIT"):
                    held = 0
                since_rebalance = 0
                qx, qy = nx, ny
            pending = None
        if i == len(data)-1 and (qx or qy):
            commission, slip = costs.execution((-qx, -qy), (px, py))
            equity -= commission+slip
            commission_total += commission
            slippage_total += slip
            trades.append({"date": date, "signal_date": date, "event": "FINAL_EXIT",
                           "delta_qx": -qx, "delta_qy": -qy, "qx": 0, "qy": 0,
                           "px": px, "py": py, "commission": commission, "slippage": slip})
            qx = qy = 0
        peak = max(peak, equity)
        if equity/peak-1 <= -.05:
            stopped = True
        tx, ty, threshold, fraction = desired_position(equity, px, py, row, params, costs, cap)
        if qx or qy:
            held += 1
            since_rebalance += 1
            direction = np.sign(qy)
            invalid = (not np.isfinite(row.prediction) or row.correlation < params.minimum_correlation
                       or row.volatility_ratio > 2.5)
            if stopped or held >= params.horizon or invalid or direction*row.prediction <= 0:
                pending = ("EXIT", 0, 0, date)
            elif since_rebalance >= 5:
                gross = abs(qx)*px+abs(qy)*py
                current_a = abs(qx)*px/gross
                target_a = row.weight_x
                nx, ny = -int(gross*target_a/px)*int(direction), int(gross*(1-target_a)/py)*int(direction)
                fee = sum(costs.execution((nx-qx, ny-qy), (px, py)))
                def variance(a, vx=row.var_x, vy=row.var_y, cov=row.covariance):
                    return a*a*vx+(1-a)**2*vy-2*a*(1-a)*cov
                risk_benefit = .5*50*gross*max(variance(current_a)-variance(target_a), 0)*params.horizon
                if (abs(current_a-target_a) > params.rebalance_band and nx and ny
                        and risk_benefit > fee and abs(nx)*px+abs(ny)*py <= equity*cap):
                    pending = ("REBALANCE", nx, ny, date)
        elif not stopped and i < len(data)-2 and tx and ty:
            pending = ("ENTRY", tx, ty, date)
        records.append({"date": date, "equity": equity, "qx": qx, "qy": qy,
                        "gross": abs(qx)*px+abs(qy)*py, "net": qx*px+qy*py,
                        "prediction": row.prediction, "entry_threshold": threshold,
                        "target_weight_x": row.weight_x, "target_weight_y": row.weight_y,
                        "desired_fraction": fraction, "stopped": stopped})
        previous = date, px, py
    curve = pd.DataFrame(records).set_index("date")
    returns = curve.equity.pct_change().fillna(0)
    vol = float(returns.std()*np.sqrt(252))
    summary = {"return": equity/capital-1, "final_equity": equity,
               "drawdown": float((curve.equity/curve.equity.cummax()-1).min()),
               "sharpe": float(returns.mean()*252/vol) if vol else 0.,
               "entries": sum(t["event"] == "ENTRY" for t in trades),
               "rebalances": sum(t["event"] == "REBALANCE" for t in trades),
               "commission": commission_total, "slippage": slippage_total,
               "borrow": borrow_total, "dividends_net": dividends_total,
               "stopped": stopped, "max_gross": float(curve.gross.max())}
    trade_columns = ["date", "signal_date", "event", "delta_qx", "delta_qy", "qx", "qy",
                     "px", "py", "commission", "slippage"]
    return summary, curve, pd.DataFrame(trades, columns=trade_columns)
