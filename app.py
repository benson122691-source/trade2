"""Streamlit dashboard for cross-asset investment simulations."""

from __future__ import annotations

from datetime import date, timedelta
from math import exp, isfinite, log1p

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from model import forecast_return


st.set_page_config(
    page_title="跨資產投資模擬器",
    page_icon="📈",
    layout="wide",
)

ASSETS = {
    "台股 ETF｜元大台灣50（0050）": {
        "ticker": "0050.TW",
        "currency": "TWD",
        "color": "#0f766e",
    },
    "美股 ETF｜SPDR S&P 500（SPY）": {
        "ticker": "SPY",
        "currency": "USD",
        "color": "#3b82f6",
    },
    "比特幣（BTC）": {
        "ticker": "BTC-USD",
        "currency": "USD",
        "color": "#d99a00",
    },
    "黃金 ETF｜SPDR Gold Shares（GLD）": {
        "ticker": "GLD",
        "currency": "USD",
        "color": "#a16207",
    },
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(start: date, end: date) -> pd.DataFrame:
    """Download adjusted closes and convert USD assets to TWD on trading dates."""
    symbols = [asset["ticker"] for asset in ASSETS.values()] + ["TWD=X"]
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
        raise ValueError("資料來源沒有回傳資料。請稍後重試或縮短日期範圍。")

    if isinstance(raw.columns, pd.MultiIndex):
        close_level = next(
            (level for level in range(raw.columns.nlevels)
             if "Close" in raw.columns.get_level_values(level)),
            None,
        )
        if close_level is None:
            raise ValueError("資料來源缺少收盤價欄位。")
        close = raw.xs("Close", axis=1, level=close_level)
        if isinstance(close, pd.Series):
            close = close.to_frame(name=symbols[0])
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        raise ValueError("資料來源缺少收盤價欄位。")

    close = close.sort_index()
    if "TWD=X" not in close.columns:
        raise ValueError("匯率資料缺漏，暫時無法將海外資產換算為新台幣。")
    fx = pd.to_numeric(close["TWD=X"], errors="coerce").dropna().sort_index()
    if fx.empty:
        raise ValueError("沒有可用的 USD/TWD 匯率資料。")

    prices: dict[str, pd.Series] = {}
    for asset in ASSETS.values():
        ticker = asset["ticker"]
        if ticker not in close.columns:
            continue
        local_price = pd.to_numeric(close[ticker], errors="coerce").dropna().sort_index()
        if local_price.empty:
            continue
        if asset["currency"] == "USD":
            aligned_fx = fx.reindex(local_price.index).ffill()
            converted = local_price.mul(aligned_fx).dropna()
            prices[ticker] = converted
        else:
            prices[ticker] = local_price

    if not prices:
        raise ValueError("所選日期範圍沒有可用的標的價格。")
    # Keep each asset's real observation calendar. Filling weekends and holidays
    # with stale prices would add artificial zero-return observations.
    return pd.DataFrame(prices).sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(
    price: pd.Series,
    horizon: int,
    periods_per_year: int,
) -> dict[str, float | int | str]:
    """Cache model comparisons for the same price series and horizon."""
    return forecast_return(price, horizon, periods_per_year=periods_per_year).to_dict()


def money_weighted_annual_return(
    cash_flows: list[tuple[pd.Timestamp, float]],
) -> float | None:
    """Solve the annual XIRR for dated contributions and terminal value."""
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
    low_value = npv(low)
    high_value = npv(high)
    while high_value > 0 and high < 1_000_000:
        high = high * 2 + 1
        high_value = npv(high)
    if not (isfinite(low_value) and isfinite(high_value)) or low_value < 0 or high_value > 0:
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
    """Simulate one-time investing or a fixed total monthly contribution budget."""
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
            "資產價值": value.astype(float),
            "累計投入": invested.astype(float),
            # This index chains asset returns and is unaffected by new cash flows.
            "報酬指數": return_index.astype(float),
        },
        index=price.index,
    )


def investment_cash_flows(
    price: pd.Series,
    principal: float,
    monthly: bool,
    final_value: float,
) -> list[tuple[pd.Timestamp, float]]:
    """Build dated cash flows for an investor-level annualized return."""
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


st.title("跨資產投資模擬器")
st.caption(
    "比較代表性資產的歷史走勢與模型情境。每個標的都以單獨投入全部預算計算，結果不是混合投資組合。"
)

with st.sidebar:
    st.header("模擬設定")
    principal = st.number_input(
        "總投入金額（新台幣）",
        min_value=1_000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
    )
    years = st.slider("歷史資料期間（年）", min_value=2, max_value=10, value=5)
    horizon = st.select_slider(
        "預估期間",
        options=[5, 10, 20, 60, 120, 252],
        value=20,
        format_func=lambda days: f"{days} 筆有效價格觀測",
    )
    method = st.radio("投入方式", ["一次投入", "定期定額"])
    selected = st.multiselect("比較標的", list(ASSETS), default=list(ASSETS))
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years + 15)
    st.caption("資料：Yahoo Finance。美股、比特幣和黃金 ETF 依交易日匯率換算為新台幣。")

if not selected:
    st.info("請在左側選取至少一個標的。")
    st.stop()

try:
    with st.spinner("正在下載市場資料並計算結果…"):
        prices = get_market_data(start_date, end_date)
except Exception as exc:
    st.error(f"無法取得市場資料：{exc}")
    st.info("請確認網路連線，或稍後重試。此原型需要 Yahoo Finance 提供歷史價格與匯率資料。")
    st.stop()

labels_by_ticker = {asset["ticker"]: label for label, asset in ASSETS.items()}
chosen_tickers = [ASSETS[label]["ticker"] for label in selected if ASSETS[label]["ticker"] in prices]
if not chosen_tickers:
    st.error("所選標的目前沒有可用資料。")
    st.stop()

st.subheader("歷史回測")
st.caption(
    "各標的使用自己的有效交易日區間。年化報酬是投資人現金流報酬率；最大回撤則依標的報酬指數計算，不受定期投入金額影響。"
)

history_frames: list[pd.DataFrame] = []
metric_cols = st.columns(min(len(chosen_tickers), 4))
for col, ticker in zip(metric_cols, chosen_tickers):
    series = prices[ticker].dropna()
    simulation = backtest(series, float(principal), method == "定期定額")
    label = labels_by_ticker[ticker]
    history_frames.append(
        pd.DataFrame(
            {
                "日期": simulation.index,
                "資產價值（元）": simulation["資產價值"].values,
                "標的": label,
            }
        )
    )
    final_value = float(simulation["資產價值"].iloc[-1])
    total_invested = float(simulation["累計投入"].iloc[-1])
    net_profit = final_value - total_invested
    returns = series.pct_change(fill_method=None).dropna()
    periods_per_year = 365 if ticker == "BTC-USD" else 252
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    return_index = simulation["報酬指數"]
    max_drawdown = float((return_index / return_index.cummax() - 1).min())
    annualized_return = money_weighted_annual_return(
        investment_cash_flows(
            series,
            float(principal),
            method == "定期定額",
            final_value,
        )
    )
    with col:
        st.metric(label, f"NT$ {final_value:,.0f}", f"損益 NT$ {net_profit:+,.0f}")
        annualized_label = (
            f"年化報酬 {annualized_return:+.1%}"
            if annualized_return is not None
            else "年化報酬無法計算"
        )
        st.caption(
            f"投入 NT$ {total_invested:,.0f}｜{annualized_label}\n\n"
            f"年化波動 {annual_volatility:.1%}｜標的最大回撤 {max_drawdown:.1%}\n\n"
            f"資料區間 {series.index[0].date()} 至 {series.index[-1].date()}"
        )

history = pd.concat(history_frames, ignore_index=True)
history_fig = px.line(
    history,
    x="日期",
    y="資產價值（元）",
    color="標的",
    color_discrete_map={label: ASSETS[label]["color"] for label in selected},
    title="各標的投資帳戶價值",
)
history_fig.update_layout(
    legend_title_text="標的",
    hovermode="x unified",
    yaxis_tickprefix="NT$ ",
    margin={"t": 55, "r": 18, "b": 18, "l": 18},
)
st.plotly_chart(history_fig, use_container_width=True)
if method == "定期定額":
    st.caption("定期定額將總預算平均分配到每月第一個有交易資料的日期。年化報酬按投入與期末資產價值的日期計算。")

st.subheader("模型估算")
st.caption(
    f"Random Forest 與 Ridge 先依訓練期內的時間順序驗證 MAE 選模，再以保留測試期比較表現。預測期為 {horizon} 個有效價格觀測。"
)
forecast_rows: list[dict[str, float | int | str]] = []
forecast_plot_rows: list[dict[str, float | str]] = []
for ticker in chosen_tickers:
    series = prices[ticker].dropna()
    label = labels_by_ticker[ticker]
    try:
        periods_per_year = 365 if ticker == "BTC-USD" else 252
        estimate = get_forecast(series, horizon, periods_per_year)
        central = float(principal) * (1 + float(estimate["prediction"]))
        lower = float(principal) * (1 + float(estimate["lower"]))
        upper = float(principal) * (1 + float(estimate["upper"]))
        forecast_rows.append(
            {
                "標的": label,
                "選用模型": estimate["selected_model"],
                "模型估計報酬": float(estimate["prediction"]) * 100,
                "歷史情境下界": float(estimate["lower"]) * 100,
                "歷史情境上界": float(estimate["upper"]) * 100,
                "測試 MAE": float(estimate["mae"]) * 100,
                "測試 RMSE": float(estimate["rmse"]) * 100,
                "零報酬基準 MAE": float(estimate["zero_baseline_mae"]) * 100,
                "訓練平均基準 MAE": float(estimate["mean_baseline_mae"]) * 100,
                "期末金額中央估算": central,
                "期末金額歷史情境": f"NT$ {lower:,.0f} 至 {upper:,.0f}",
            }
        )
        forecast_plot_rows.append(
            {"標的": label, "lower": lower, "central": central, "upper": upper}
        )
    except Exception as exc:
        st.warning(f"{label}：{exc}")

if forecast_rows:
    forecast = pd.DataFrame(forecast_rows)
    st.dataframe(
        forecast,
        hide_index=True,
        use_container_width=True,
        column_config={
            "模型估計報酬": st.column_config.NumberColumn(format="%.2f%%"),
            "歷史情境下界": st.column_config.NumberColumn(format="%.2f%%"),
            "歷史情境上界": st.column_config.NumberColumn(format="%.2f%%"),
            "測試 MAE": st.column_config.NumberColumn(format="%.2f%%"),
            "測試 RMSE": st.column_config.NumberColumn(format="%.2f%%"),
            "零報酬基準 MAE": st.column_config.NumberColumn(format="%.2f%%"),
            "訓練平均基準 MAE": st.column_config.NumberColumn(format="%.2f%%"),
            "期末金額中央估算": st.column_config.NumberColumn(format="NT$ %.0f"),
        },
    )
    st.caption("模型誤差以對數報酬計算，表格用百分比格式顯示；模型估計與歷史情境都假設全部預算現在投入單一標的。")

    st.markdown("**預估期末金額**")
    forecast_fig = go.Figure()
    for row in forecast_plot_rows:
        forecast_fig.add_trace(
            go.Scatter(
                x=[row["lower"], row["upper"]],
                y=[row["標的"], row["標的"]],
                mode="lines+markers",
                name=f"{row['標的']} 歷史情境",
                line={"color": "#94a3b8", "width": 5},
                marker={"size": 8, "color": "#94a3b8"},
                hovertemplate=f"{row['標的']}<br>測試期 10% 至 90% 分位數：NT$ {row['lower']:,.0f} 至 {row['upper']:,.0f}<extra></extra>",
            )
        )
        forecast_fig.add_trace(
            go.Scatter(
                x=[row["central"]],
                y=[row["標的"]],
                mode="markers+text",
                name=f"{row['標的']} 中央估算",
                text=[f"NT$ {row['central']:,.0f}"],
                textposition="middle right",
                marker={"size": 12, "color": "#0f766e"},
                hovertemplate=f"{row['標的']}<br>模型中央估算：NT$ {row['central']:,.0f}<extra></extra>",
            )
        )
    forecast_fig.update_layout(
        height=max(260, 68 * len(forecast_plot_rows)),
        showlegend=False,
        hovermode="closest",
        xaxis_title="期末金額（新台幣）",
        yaxis_title="",
        margin={"l": 10, "r": 70, "t": 16, "b": 42},
    )
    st.plotly_chart(forecast_fig, use_container_width=True)

with st.expander("計算方法與限制"):
    st.markdown(
        """
- 歷史價格使用 Yahoo Finance 的調整後價格；海外資產依 USD/TWD 匯率換算。
- 週末和休市日不補上前一日的資產價格，避免把重複價格算成零報酬交易日。
- 一次投入在期間起點投入全部預算；定期定額把同一筆總預算平均分配到每月第一個有效交易日。
- 年化報酬依實際投入日期和期末資產價值求解 XIRR。波動度以有效觀測日報酬年化；最大回撤以標的報酬指數計算，因此不會被後續投入金額抬高。
- 每一個標的都個別使用全部預算估算，不構成資產配置組合。模型期末估算也假設目前一次投入。
- 模型情境範圍來自保留測試期實際報酬的第 10 與第 90 百分位數，不是信賴區間，也不保證未來落在範圍內。
- MAE、RMSE 以對數報酬誤差計算。每日預測標籤的持有期間可能重疊，測試誤差有時間相依性，應作描述性參考。
- 尚未計入手續費、稅負、滑價、配息稅務或匯兌成本。資料可能缺漏、延遲或修訂；本系統僅供教育與方法比較，不提供個人化投資建議。
        """
    )
