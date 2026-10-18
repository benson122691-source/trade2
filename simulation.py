"""Market data, backtesting, and forecast assembly for the web interface."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from math import exp, isfinite, log1p

import numpy as np
import pandas as pd
import yfinance as yf

from model import forecast_return


ASSETS = {
    "0050.TW": {
        "ticker": "0050.TW",
        "name": "台股 ETF｜元大台灣 50",
        "currency": "TWD",
        "color": "#147d72",
    },
    "SPY": {
        "ticker": "SPY",
        "name": "美股 ETF｜S&P 500",
        "currency": "USD",
        "color": "#477ca8",
    },
    "BTC-USD": {
        "ticker": "BTC-USD",
        "name": "比特幣",
        "currency": "USD",
        "color": "#d99a00",
    },
    "GLD": {
        "ticker": "GLD",
        "name": "黃金 ETF",
        "currency": "USD",
        "color": "#a16207",
    },
}


@lru_cache(maxsize=8)
def get_market_data(start: date, end: date) -> pd.DataFrame:
    """Download adjusted closes and convert USD assets to TWD."""
    symbols = list(ASSETS) + ["TWD=X"]
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("資料來源沒有回傳行情。請稍後重試或調整日期範圍。")

    if isinstance(raw.columns, pd.MultiIndex):
        close_level = next(
            (
                level
                for level in range(raw.columns.nlevels)
                if "Close" in raw.columns.get_level_values(level)
            ),
            None,
        )
        if close_level is None:
            raise ValueError("資料來源缺少收盤價欄位。")
        close = raw.xs("Close", axis=1, level=close_level)
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        raise ValueError("資料來源缺少收盤價欄位。")

    close = close.sort_index()
    if "TWD=X" not in close.columns:
        raise ValueError("匯率資料缺漏，暫時無法換算海外資產。")
    fx = pd.to_numeric(close["TWD=X"], errors="coerce").dropna().sort_index()
    if fx.empty:
        raise ValueError("沒有可用的 USD/TWD 匯率資料。")

    prices: dict[str, pd.Series] = {}
    for ticker, asset in ASSETS.items():
        if ticker not in close.columns:
            continue
        local_price = (
            pd.to_numeric(close[ticker], errors="coerce").dropna().sort_index()
        )
        if local_price.empty:
            continue
        if asset["currency"] == "USD":
            aligned_fx = fx.reindex(local_price.index).ffill()
            prices[ticker] = local_price.mul(aligned_fx).dropna()
        else:
            prices[ticker] = local_price

    if not prices:
        raise ValueError("所選日期範圍沒有可用的標的價格。")
    # Keep real observation dates; do not fill weekends or market holidays.
    return pd.DataFrame(prices).sort_index()


def money_weighted_annual_return(
    cash_flows: list[tuple[pd.Timestamp, float]],
) -> float | None:
    """Solve XIRR for dated contributions and the ending account value."""
    if len(cash_flows) < 2:
        return None
    start = min(day for day, _ in cash_flows)
    dated_flows = [((day - start).days / 365.25, amount) for day, amount in cash_flows]

    def npv(rate: float) -> float:
        log_growth = log1p(rate)
        try:
            return sum(amount * exp(-log_growth * years) for years, amount in dated_flows)
        except OverflowError:
            return float("inf")

    low, high = -0.9999, 10.0
    low_value, high_value = npv(low), npv(high)
    while high_value > 0 and high < 1_000_000:
        high = high * 2 + 1
        high_value = npv(high)
    if (
        not (isfinite(low_value) and isfinite(high_value))
        or low_value < 0
        or high_value > 0
    ):
        return None
    for _ in range(120):
        middle = (low + high) / 2
        middle_value = npv(middle)
        if abs(middle_value) < 1e-7:
            return middle
        if (low_value > 0) == (middle_value > 0):
            low, low_value = middle, middle_value
        else:
            high = middle
    return (low + high) / 2


def backtest(price: pd.Series, principal: float, monthly: bool) -> pd.DataFrame:
    """Simulate one-time investing or monthly investments from one budget."""
    price = price.dropna().astype(float).sort_index()
    if price.empty or (price <= 0).any():
        raise ValueError("回測價格必須是正數，且至少要有一筆資料。")

    returns = price.pct_change(fill_method=None).fillna(0.0)
    return_index = (1 + returns).cumprod()
    if not monthly:
        invested = pd.Series(float(principal), index=price.index)
        value = float(principal) * price / float(price.iloc[0])
    else:
        month_groups = pd.Series(np.arange(len(price)), index=price.index).groupby(
            price.index.to_period("M")
        )
        contribution_dates = [price.index[int(group.iloc[0])] for _, group in month_groups]
        contribution = float(principal) / max(len(contribution_dates), 1)
        units = 0.0
        contribution_count = 0
        values: list[float] = []
        invested_values: list[float] = []
        next_date_i = 0
        for dt, market_price in price.items():
            if next_date_i < len(contribution_dates) and dt == contribution_dates[next_date_i]:
                units += contribution / market_price
                contribution_count += 1
                next_date_i += 1
            values.append(units * market_price)
            invested_values.append(contribution * contribution_count)
        value = pd.Series(values, index=price.index)
        invested = pd.Series(invested_values, index=price.index)

    return pd.DataFrame(
        {
            "asset_value": value.astype(float),
            "total_invested": invested.astype(float),
            "return_index": return_index.astype(float),
        },
        index=price.index,
    )


def investment_cash_flows(
    price: pd.Series,
    principal: float,
    monthly: bool,
    final_value: float,
) -> list[tuple[pd.Timestamp, float]]:
    """Build the dated investment cash flows used by XIRR."""
    if not monthly:
        return [(price.index[0], -float(principal)), (price.index[-1], final_value)]
    month_groups = pd.Series(np.arange(len(price)), index=price.index).groupby(
        price.index.to_period("M")
    )
    dates = [price.index[int(group.iloc[0])] for _, group in month_groups]
    contribution = float(principal) / max(len(dates), 1)
    flows = [(dt, -contribution) for dt in dates]
    flows.append((price.index[-1], final_value))
    return flows


def run_simulation(
    *,
    start: date,
    end: date,
    principal: float,
    method: str,
    selected: list[str],
    horizon: int,
) -> dict:
    """Create JSON-ready history, risk metrics, and model scenarios."""
    unknown = [ticker for ticker in selected if ticker not in ASSETS]
    if unknown:
        raise ValueError("比較標的清單包含無效項目。")

    prices = get_market_data(start, end)
    histories: list[dict] = []
    metrics: list[dict] = []
    forecasts: list[dict] = []
    warnings: list[str] = []

    for ticker in dict.fromkeys(selected):
        if ticker not in prices:
            warnings.append(f"{ASSETS[ticker]['name']} 在這段期間沒有可用價格資料。")
            continue
        series = prices[ticker].dropna().astype(float).sort_index()
        if series.empty:
            warnings.append(f"{ASSETS[ticker]['name']} 在這段期間沒有可用價格資料。")
            continue

        asset = ASSETS[ticker]
        simulation = backtest(series, principal, method == "monthly")
        final_value = float(simulation["asset_value"].iloc[-1])
        total_invested = float(simulation["total_invested"].iloc[-1])
        returns = series.pct_change(fill_method=None).dropna()
        periods_per_year = 365 if ticker == "BTC-USD" else 252
        volatility = (
            float(returns.std(ddof=1) * np.sqrt(periods_per_year))
            if len(returns) > 1
            else 0.0
        )
        return_index = simulation["return_index"]
        max_drawdown = float((return_index / return_index.cummax() - 1).min())
        annual_return = money_weighted_annual_return(
            investment_cash_flows(
                series, principal, method == "monthly", final_value
            )
        )

        histories.extend(
            {
                "ticker": ticker,
                "date": day.strftime("%Y-%m-%d"),
                "value": float(value),
            }
            for day, value in simulation["asset_value"].items()
        )
        metrics.append(
            {
                "ticker": ticker,
                "name": asset["name"],
                "color": asset["color"],
                "final_value": final_value,
                "total_invested": total_invested,
                "profit": final_value - total_invested,
                "annual_return": annual_return,
                "annual_volatility": volatility,
                "max_drawdown": max_drawdown,
                "data_start": series.index[0].strftime("%Y-%m-%d"),
                "data_end": series.index[-1].strftime("%Y-%m-%d"),
            }
        )

        try:
            estimate = forecast_return(
                series,
                horizon,
                periods_per_year=periods_per_year,
            ).to_dict()
            forecasts.append(
                {
                    "ticker": ticker,
                    "name": asset["name"],
                    "model": estimate["selected_model"],
                    "prediction": estimate["prediction"],
                    "lower": estimate["lower"],
                    "upper": estimate["upper"],
                    "mae": estimate["mae"],
                    "rmse": estimate["rmse"],
                    "zero_baseline_mae": estimate["zero_baseline_mae"],
                    "mean_baseline_mae": estimate["mean_baseline_mae"],
                    "central_value": principal * (1 + estimate["prediction"]),
                    "lower_value": principal * (1 + estimate["lower"]),
                    "upper_value": principal * (1 + estimate["upper"]),
                }
            )
        except Exception as exc:
            warnings.append(f"{asset['name']}：模型資料不足或估算失敗（{exc}）")

    if not metrics:
        raise ValueError("所選標的在這段期間沒有可用資料，請調整日期範圍。")

    return {
        "method": method,
        "principal": principal,
        "horizon": horizon,
        "histories": histories,
        "metrics": metrics,
        "forecasts": forecasts,
        "warnings": warnings,
    }
