"""Jesse Regime Momentum Strategy with Variance-Ratio & ATR Position Sizing.

Implements a dual-EMA trend and variance-ratio persistence regime filter
in the Jesse event-driven backtesting framework, enforcing strict 1.0% account
equity risk per trade (`utils.risk_to_qty`), scaled take-profit legs, and
trailing ATR stop updates in `update_position()`.
"""

from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
import numpy as np


class JesseRegimeMomentum(Strategy):
    """Event-driven trend strategy with 1.0% equity risk sizing."""

    def _is_persistent_regime(self) -> bool:
        """Variance-ratio check (> 1.05 indicates trending serial persistence)."""
        closes = self.candles[-60:, 2]
        if len(closes) < 60:
            return False
        ret_1 = np.diff(np.log(closes))
        ret_4 = np.log(closes[4:] / closes[:-4])
        var_1 = float(np.var(ret_1))
        var_4 = float(np.var(ret_4))
        vr = var_4 / max(4.0 * var_1, 1e-12)
        return vr > 1.05

    @property
    def fast_ema(self) -> float:
        return float(ta.ema(self.candles, period=21))

    @property
    def slow_ema(self) -> float:
        return float(ta.ema(self.candles, period=55))

    @property
    def atr(self) -> float:
        return float(ta.atr(self.candles, period=14))

    @property
    def adx(self) -> float:
        return float(ta.adx(self.candles, period=14))

    def should_long(self) -> bool:
        return (
            self.fast_ema > self.slow_ema
            and self.close > self.fast_ema
            and self.adx > 22.0
            and self._is_persistent_regime()
        )

    def should_short(self) -> bool:
        return (
            self.fast_ema < self.slow_ema
            and self.close < self.fast_ema
            and self.adx > 22.0
            and self._is_persistent_regime()
        )

    def should_cancel_entry(self) -> bool:
        return True

    def go_long(self) -> None:
        stop_price = self.close - 2.0 * self.atr
        qty = utils.risk_to_qty(self.balance, 1.0, self.close, stop_price, fee_rate=self.fee_rate)
        self.buy = qty, self.close
        self.stop_loss = qty, stop_price
        self.take_profit = [
            (qty * 0.5, self.close + 2.5 * self.atr),
            (qty * 0.5, self.close + 4.5 * self.atr),
        ]

    def go_short(self) -> None:
        stop_price = self.close + 2.0 * self.atr
        qty = utils.risk_to_qty(self.balance, 1.0, self.close, stop_price, fee_rate=self.fee_rate)
        self.sell = qty, self.close
        self.stop_loss = qty, stop_price
        self.take_profit = [
            (qty * 0.5, self.close - 2.5 * self.atr),
            (qty * 0.5, self.close - 4.5 * self.atr),
        ]

    def update_position(self) -> None:
        """Trail stop loss by 2.0x ATR once trade moves into +1.5x ATR profit."""
        if self.is_long and self.close > self.position.entry_price + 1.5 * self.atr:
            new_stop = max(self.position.entry_price, self.close - 2.0 * self.atr)
            self.stop_loss = self.position.qty, new_stop
        elif self.is_short and self.close < self.position.entry_price - 1.5 * self.atr:
            new_stop = min(self.position.entry_price, self.close + 2.0 * self.atr)
            self.stop_loss = self.position.qty, new_stop
