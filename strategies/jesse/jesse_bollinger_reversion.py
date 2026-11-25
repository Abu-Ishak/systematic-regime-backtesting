"""Jesse Multi-Timeframe Bollinger Reversion Strategy with Volatility Capping.

Uses a 4h EMA-200 structural trend anchor combined with 15m Bollinger Band %B
and RSI exhaustion triggers, capping position risk at 0.75% of equity and
enforcing a 32-bar maximum holding-period time stop in `update_position()`.
"""

from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils


class JesseBollingerReversion(Strategy):
    """Multi-timeframe volatility reversion strategy with time-stop discipline."""

    max_hold_bars: int = 32

    @property
    def anchor_ema_4h(self) -> float:
        candles_4h = self.get_candles(self.exchange, self.symbol, "4h")
        return float(ta.ema(candles_4h, period=200))

    @property
    def bb(self):
        return ta.bollinger_bands(self.candles, period=20, devup=2.2, devdn=2.2)

    @property
    def rsi(self) -> float:
        return float(ta.rsi(self.candles, period=14))

    @property
    def atr(self) -> float:
        return float(ta.atr(self.candles, period=14))

    def should_long(self) -> bool:
        return (
            self.close > self.anchor_ema_4h
            and self.close < self.bb.lowerband
            and self.rsi < 32.0
        )

    def should_short(self) -> bool:
        return (
            self.close < self.anchor_ema_4h
            and self.close > self.bb.upperband
            and self.rsi > 68.0
        )

    def should_cancel_entry(self) -> bool:
        return True

    def go_long(self) -> None:
        stop_price = self.close - 1.8 * self.atr
        qty = utils.risk_to_qty(self.balance, 0.75, self.close, stop_price, fee_rate=self.fee_rate)
        self.buy = qty, self.close
        self.stop_loss = qty, stop_price
        self.take_profit = qty, float(self.bb.middleband)

    def go_short(self) -> None:
        stop_price = self.close + 1.8 * self.atr
        qty = utils.risk_to_qty(self.balance, 0.75, self.close, stop_price, fee_rate=self.fee_rate)
        self.sell = qty, self.close
        self.stop_loss = qty, stop_price
        self.take_profit = qty, float(self.bb.middleband)

    def update_position(self) -> None:
        bars_held = int((self.current_candle[0] - self.position.opened_at) / (15 * 60 * 1000))
        if bars_held >= self.max_hold_bars:
            self.liquidate()
