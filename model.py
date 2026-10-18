"""Machine-learning model for the cross-asset investment simulator.

The model forecasts the H-observation log return from price-derived features.
It uses a chronological holdout and purges overlapping labels at the train/test
boundary. The holdout metrics are diagnostics, not evidence of future profits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "volatility_60",
    "price_vs_ma_20",
    "price_vs_ma_60",
)


@dataclass(frozen=True)
class ForecastResult:
    """Forecast, empirical scenario range, and chronological holdout metrics."""

    prediction: float
    lower: float
    upper: float
    selected_model: str
    mae: float
    rmse: float
    random_forest_mae: float
    random_forest_rmse: float
    ridge_mae: float
    ridge_rmse: float
    zero_baseline_mae: float
    mean_baseline_mae: float
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _clean_prices(prices: pd.Series) -> pd.Series:
    series = pd.to_numeric(prices, errors="coerce").astype(float).sort_index()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series > 0]
    if not series.index.is_unique:
        series = series.groupby(level=0).last()
    if len(series) < 130:
        raise ValueError("至少需要約 130 筆有效日價格，才能建立和評估模型。")
    return series


def build_features(prices: pd.Series, periods_per_year: int = 252) -> pd.DataFrame:
    """Create price-only features available at the forecast date."""
    if periods_per_year < 1:
        raise ValueError("periods_per_year 必須至少為 1。")
    price = _clean_prices(prices)
    daily_returns = price.pct_change(fill_method=None)
    features = pd.DataFrame(index=price.index)
    features["momentum_5"] = price.pct_change(5, fill_method=None)
    features["momentum_20"] = price.pct_change(20, fill_method=None)
    features["momentum_60"] = price.pct_change(60, fill_method=None)
    features["volatility_20"] = daily_returns.rolling(20).std() * np.sqrt(periods_per_year)
    features["volatility_60"] = daily_returns.rolling(60).std() * np.sqrt(periods_per_year)
    features["price_vs_ma_20"] = price / price.rolling(20).mean() - 1
    features["price_vs_ma_60"] = price / price.rolling(60).mean() - 1
    return features.replace([np.inf, -np.inf], np.nan)


def _make_supervised(
    prices: pd.Series,
    horizon: int,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    if horizon < 1:
        raise ValueError("horizon 必須至少為 1 個有效價格觀測。")
    price = _clean_prices(prices)
    features = build_features(price, periods_per_year=periods_per_year)
    # At observation t, label the log return from t through t+horizon.
    target = np.log(price.shift(-horizon) / price).rename("target_log_return")
    joined = features.join(target).replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise ValueError("歷史資料不足以建立指定預測期間的訓練資料。")
    return joined.loc[:, list(FEATURE_COLUMNS)], joined["target_log_return"]


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )


def _new_ridge_model():
    """Scaled Ridge regression provides a compact, interpretable benchmark."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def forecast_return(
    prices: pd.Series,
    horizon: int,
    periods_per_year: int = 252,
) -> ForecastResult:
    """Forecast the next H-day return and evaluate on a later chronological holdout.

    The test-set 10th and 90th percentiles form an empirical scenario range.
    They are not a confidence interval or a guarantee that future returns will
    stay within the range.
    """
    price = _clean_prices(prices)
    features, target = _make_supervised(price, horizon, periods_per_year)
    n_rows = len(features)
    split = int(n_rows * 0.8)

    # Remove labels whose forward-return windows would overlap the test period.
    train_end = split - horizon
    train_x, train_y = features.iloc[:train_end], target.iloc[:train_end]
    test_x, test_y = features.iloc[split:], target.iloc[split:]
    if len(train_x) < 60 or len(test_x) < 20:
        raise ValueError(
            "資料不足以同時保留訓練區間、預測期間隔離區和測試區。"
            "請增加歷史資料或縮短預測天數。"
        )

    # Select a model using a later validation slice inside the training data.
    # Keep the final test slice untouched until model choice is complete.
    validation_split = int(len(train_x) * 0.8)
    validation_train_end = validation_split - horizon
    if validation_train_end >= 60 and len(train_x) - validation_split >= 20:
        fit_x = train_x.iloc[:validation_train_end]
        fit_y = train_y.iloc[:validation_train_end]
        validation_x = train_x.iloc[validation_split:]
        validation_y = train_y.iloc[validation_split:]
        rf_candidate = _new_model().fit(fit_x, fit_y)
        ridge_candidate = _new_ridge_model().fit(fit_x, fit_y)
        rf_validation_mae = mean_absolute_error(validation_y, rf_candidate.predict(validation_x))
        ridge_validation_mae = mean_absolute_error(validation_y, ridge_candidate.predict(validation_x))
        selected_model = "Random Forest" if rf_validation_mae <= ridge_validation_mae else "Ridge"
    else:
        # With too little data for a second purged split, keep the model choice
        # fixed in advance. Do not choose a winner based on the final test set.
        selected_model = "Random Forest"

    rf_evaluation_model = _new_model().fit(train_x, train_y)
    rf_test_prediction = rf_evaluation_model.predict(test_x)
    ridge_evaluation_model = _new_ridge_model().fit(train_x, train_y)
    ridge_test_prediction = ridge_evaluation_model.predict(test_x)
    zero_prediction = np.zeros(len(test_y))
    mean_prediction = np.full(len(test_y), float(train_y.mean()))
    rf_mae = float(mean_absolute_error(test_y, rf_test_prediction))
    rf_rmse = float(np.sqrt(mean_squared_error(test_y, rf_test_prediction)))
    ridge_mae = float(mean_absolute_error(test_y, ridge_test_prediction))
    ridge_rmse = float(np.sqrt(mean_squared_error(test_y, ridge_test_prediction)))
    mae, rmse = (rf_mae, rf_rmse) if selected_model == "Random Forest" else (ridge_mae, ridge_rmse)
    zero_mae = float(mean_absolute_error(test_y, zero_prediction))
    mean_mae = float(mean_absolute_error(test_y, mean_prediction))

    # The forecast row has current features but no future label yet.
    latest_features = (
        build_features(price, periods_per_year=periods_per_year)
        .loc[:, list(FEATURE_COLUMNS)]
        .dropna()
        .iloc[[-1]]
    )
    final_model = _new_model() if selected_model == "Random Forest" else _new_ridge_model()
    final_model.fit(features, target)
    predicted_log_return = float(final_model.predict(latest_features)[0])
    lower_log, upper_log = target.iloc[split:].quantile([0.10, 0.90])

    return ForecastResult(
        prediction=float(np.expm1(predicted_log_return)),
        lower=float(np.expm1(lower_log)),
        upper=float(np.expm1(upper_log)),
        selected_model=selected_model,
        mae=mae,
        rmse=rmse,
        random_forest_mae=rf_mae,
        random_forest_rmse=rf_rmse,
        ridge_mae=ridge_mae,
        ridge_rmse=ridge_rmse,
        zero_baseline_mae=zero_mae,
        mean_baseline_mae=mean_mae,
        train_rows=len(train_x),
        test_rows=len(test_x),
    )


if __name__ == "__main__":
    # Example: replace with a CSV containing Date and Close columns.
    sample = pd.read_csv("prices.csv", parse_dates=["Date"]).set_index("Date")
    result = forecast_return(sample["Close"], horizon=20)
    print(pd.Series(result.to_dict()).to_string())
