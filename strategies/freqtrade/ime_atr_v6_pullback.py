"""Liquidity-Pullback & Volatility-Managed Systematic Strategy (Freqtrade IStrategy v3).

Refactored and sanitized institutional version of `imeatrv6.py` for systematic
research across liquid benchmark instruments. Combines:
1. Short-horizon ATR(3) intrabar liquidity displacement entries gated by higher-period EMA slope.
2. Fractional capital allocation (`custom_stake_amount`) and capped leverage (`max_leverage / 20`).
3. ATR-anchored initial stoploss with dynamic breakeven ratchet once unrealized profit
   reaches 50% of the volatility take-profit target (`custom_stoploss`).
4. Bounded partial profit de-risking and time-decay exit discipline (`adjust_trade_position`, `custom_exit`).
"""

from datetime import datetime
from typing import Optional

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    stoploss_from_absolute,
)
from pandas import DataFrame
import talib.abstract as ta


BENCHMARK_ASSETS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "BNB"]


class ImeAtrPullbackV6(IStrategy):
    """ATR pullback mean-reversion within macro EMA trend regimes."""

    INTERFACE_VERSION = 3
    can_short: bool = True
    minimal_roi = {}
    unfilledtimeout = {"entry": 30, "exit": 30}

    position_adjustment_enable = True
    use_custom_stoploss = True
    stoploss = -0.99  # Overridden immediately after fill via custom_stoploss
    trailing_stop = False

    timeframe = "15m"
    leverage_divisor = 20
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 100

    # Hyperoptable risk & regime parameters
    tpatr = DecimalParameter(low=1.0, high=3.0, default=2.0, space="sell", decimals=1, optimize=True, load=True)
    entryatr = DecimalParameter(low=1.0, high=3.0, default=2.0, space="buy", decimals=1, optimize=True, load=True)
    slatr = DecimalParameter(low=0.5, high=2.0, default=1.0, space="sell", decimals=1, optimize=True, load=True)
    ema = CategoricalParameter(["30", "40", "60", "90"], default="90", space="buy", optimize=True, load=True)
    time_out = IntParameter(low=12, high=98, default=16, space="sell", optimize=True, load=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "entryatr": {"color": "green"},
            "entryatru": {"color": "green"},
            "ema_90": {"color": "black"},
        },
        "subplots": {},
    }

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Cap effective leverage to a conservative fraction of exchange maximum."""
        return float(max(1, int(max_leverage / self.leverage_divisor)))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float:
        """Reserve 50% of per-slot capital for optional bounded risk scaling."""
        if self.config["stake_amount"] == "unlimited":
            max_trades = max(1, int(self.config.get("max_open_trades", 10)))
            return self.wallets.get_total_stake_amount() / (2.0 * max_trades)
        return proposed_stake / 2.0

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=3)
        dataframe["entryatr"] = dataframe["close"] - dataframe["atr"] * self.entryatr.value
        dataframe["entryatru"] = dataframe["close"] + dataframe["atr"] * self.entryatr.value

        for period in (30, 40, 60, 90):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        asset = metadata["pair"].split("/")[0]
        entryatr = getattr(self, f"entryatr_{asset}", self.entryatr)
        ema_col = f"ema_{self.ema.value}"

        dataframe.loc[
            (
                (dataframe["low"] < (dataframe["close"].shift(1) - dataframe["atr"].shift(1) * entryatr.value))
                & (dataframe[ema_col] > dataframe[ema_col].shift(1))
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "atr_pullback_long")

        dataframe.loc[
            (
                (dataframe["high"] > (dataframe["close"].shift(1) + dataframe["atr"].shift(1) * entryatr.value))
                & (dataframe[ema_col] < dataframe[ema_col].shift(1))
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "atr_pullback_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """Volatility-scaled take-profit, holding-duration exit, and adverse tail cut."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_bars = int((current_time - trade.open_date_utc).total_seconds() / 60 / 15)
        if trade_bars + 1 >= len(dataframe):
            return "time_out_horizon" if current_profit > 0 else None

        entry_candle = dataframe.iloc[-trade_bars - 1].squeeze()
        entry_price = trade.open_rate
        tp_pct = (entry_candle["atr"] * self.tpatr.value * trade.leverage) / entry_price
        adverse_pct = (entry_candle["atr"] * trade.leverage * 2.0 * self.entryatr.value) / entry_price

        if current_profit > tp_pct:
            return "tp_short" if trade.is_short else "tp_long"
        if trade_bars / 3.0 > self.time_out.value and current_profit > 0:
            return "time_out"
        if current_profit < -adverse_pct:
            return "adverse_volatility_cut"
        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        """Scale in once on adverse displacement or de-risk 50% at half target."""
        if trade.nr_of_successful_entries <= 1:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        trade_bars = int((current_time - trade.open_date_utc).total_seconds() / 60 / 15)
        if trade_bars + 1 >= len(dataframe):
            return None

        entry_candle = dataframe.iloc[-trade_bars - 1].squeeze()
        asset = trade.pair.split("/")[0]
        entryatr = getattr(self, f"entryatr_{asset}", self.entryatr)
        tp_pct = (entry_candle["atr"] * entryatr.value * trade.leverage) / trade.open_rate

        if current_profit < -tp_pct:
            return trade.stake_amount, "bounded_scale_in"
        if current_profit > (tp_pct / 2.0):
            return -(trade.stake_amount / 2.0), "half_profit_derisk"
        return None

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
        """Set ATR-based initial stop and ratchet stop to breakeven at +0.5x TP."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_bars = int((current_time - trade.open_date_utc).total_seconds() / 60 / 15)
        asset = pair.split("/")[0]
        entryatr = getattr(self, f"entryatr_{asset}", self.entryatr)

        if after_fill and trade.nr_of_successful_entries == 1:
            offset = last_candle["atr"] * (entryatr.value + self.slatr.value)
            stop_price = trade.open_rate + offset if trade.is_short else trade.open_rate - offset
            return stoploss_from_absolute(
                stop_price,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )

        if trade_bars + 1 < len(dataframe):
            entry_candle = dataframe.iloc[-trade_bars - 1].squeeze()
            tp_pct = (entry_candle["atr"] * entryatr.value * trade.leverage) / trade.open_rate
            if current_profit > (tp_pct / 2.0):
                return stoploss_from_absolute(
                    trade.open_rate,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )

        return 1.0


for symbol in BENCHMARK_ASSETS:
    setattr(
        ImeAtrPullbackV6,
        f"entryatr_{symbol}",
        DecimalParameter(low=1.0, high=3.0, default=2.0, space="buy", decimals=1, optimize=True, load=True),
    )
