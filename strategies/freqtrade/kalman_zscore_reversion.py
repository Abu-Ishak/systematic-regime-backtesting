"""Kalman Filter Mean-Reversion Strategy Gated by ADX Non-Trending Regime.

Estimates dynamic equilibrium fair value using a 1D scalar Kalman Filter and
enters mean-reversion positions when normalized price deviation |Z| > 2.0, strictly
under low-trend conditions (ADX < 25) to avoid fighting structural breakouts.
"""

import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


def kalman_filter_1d(series: np.ndarray, process_var: float = 1e-4, obs_var: float = 1e-2) -> np.ndarray:
    """Estimate latent equilibrium price trajectory using a scalar Kalman Filter."""
    n = len(series)
    estimate = np.zeros(n, dtype=float)
    x_hat = float(series[0]) if n > 0 else 0.0
    p_cov = 1.0

    for i in range(n):
        p_pred = p_cov + process_var
        gain = p_pred / (p_pred + obs_var)
        x_hat = x_hat + gain * (float(series[i]) - x_hat)
        p_cov = (1.0 - gain) * p_pred
        estimate[i] = x_hat
    return estimate


class KalmanZScoreReversion(IStrategy):
    """Regime-gated statistical mean-reversion strategy."""

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "15m"
    startup_candle_count: int = 80

    minimal_roi = {"0": 0.03, "60": 0.015}
    stoploss = -0.025
    use_exit_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["kf_mean"] = kalman_filter_1d(dataframe["close"].to_numpy())
        spread = dataframe["close"] - dataframe["kf_mean"]
        dataframe["spread_std"] = spread.rolling(48).std().replace(0.0, np.nan)
        dataframe["zscore"] = spread / dataframe["spread_std"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["zscore"] < -2.0)
                & (dataframe["adx"] < 25.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "kalman_oversold_z")

        dataframe.loc[
            (
                (dataframe["zscore"] > 2.0)
                & (dataframe["adx"] < 25.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "kalman_overbought_z")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["zscore"] >= -0.1, ["exit_long", "exit_tag"]] = (1, "kalman_equilibrium")
        dataframe.loc[dataframe["zscore"] <= 0.1, ["exit_short", "exit_tag"]] = (1, "kalman_equilibrium")
        return dataframe
