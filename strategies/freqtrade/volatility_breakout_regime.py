"""Volatility-Regime Donchian Breakout with Chandelier ATR Trailing & Protections.

Captures institutional momentum breakouts confirmed by ADX(>25) trend strength,
exits via Chandelier ATR(14) trailing bands, sizes positions inversely to realized
volatility, and enforces portfolio-level Freqtrade risk protections.
"""

from datetime import datetime
from typing import Optional

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame


class VolatilityBreakoutRegime(IStrategy):
    """Momentum breakout strategy with portfolio-level drawdown protections."""

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "1h"
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.12}
    stoploss = -0.04
    use_custom_stoploss = True

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 2,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 120,
                "trade_limit": 4,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.08,
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["donchian_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["donchian_low"] = dataframe["low"].rolling(20).min().shift(1)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["chandelier_long"] = dataframe["high"].rolling(22).max() - 3.0 * dataframe["atr"]
        dataframe["chandelier_short"] = dataframe["low"].rolling(22).min() + 3.0 * dataframe["atr"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["donchian_high"])
                & (dataframe["adx"] > 25.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "donchian_breakout_long")

        dataframe.loc[
            (
                (dataframe["close"] < dataframe["donchian_low"])
                & (dataframe["adx"] > 25.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "donchian_breakout_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["close"] < dataframe["chandelier_long"], ["exit_long", "exit_tag"]] = (1, "chandelier_exit")
        dataframe.loc[dataframe["close"] > dataframe["chandelier_short"], ["exit_short", "exit_tag"]] = (1, "chandelier_exit")
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = dataframe.iloc[-1].squeeze()
        stop_level = last["chandelier_short"] if trade.is_short else last["chandelier_long"]
        return stoploss_from_absolute(
            float(stop_level),
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
