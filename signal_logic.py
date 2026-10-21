"""
signal_logic.py

Converts the TCN model's predicted next-bar Open into BUY / SELL / HOLD.

Responsibility of this module (and ONLY this module):
    Given current price, indicator values (EMA9/20/50, VWAP, RSI14, MACD),
    and predicted_next_open (from the TCN), decide BUY / SELL / HOLD.

This module NEVER trains or calls the TCN. It only consumes the
prediction the model already produced. This keeps the trading logic
fully swappable without ever touching the trained model.

Decision path
--------------
    1. predicted_move_pct = (predicted_next_open - current_price) / current_price
    2. trend             = "UP" if ema9 > ema20, "DOWN" if ema9 < ema20, else "FLAT"
       (this "trend" column is kept for backwards-compat / charting; the
       actual BUY/SELL gate below uses the fuller indicator confirmation)
    3. The TCN prediction is treated as the PRIMARY directional signal.
       Indicators (EMA9/20/50, VWAP, RSI14, MACD) are CONFIRMATION filters
       on top of it:

       BUY  requires ALL of:
        #    - predicted_move_pct >  threshold_pct   (TCN bullish)
           - ema9  > ema20 > ema50                 (EMA stack bullish)
        # #    - current_price > vwap                  (above VWAP)
        #    - macd_line > macd_signal AND macd_hist > 0   (MACD bullish)
        #    - 50 < rsi14 < 70                        (bullish momentum,
             not already overbought-extreme)

       SELL requires ALL of:
           - predicted_move_pct < -threshold_pct   (TCN bearish)
           - ema9  < ema20 < ema50                 (EMA stack bearish)
           - current_price < vwap                  (below VWAP)
           - macd_line < macd_signal AND macd_hist < 0   (MACD bearish)
           - 30 < rsi14 < 50                        (bearish momentum,
             not already oversold-extreme)

       HOLD otherwise -- i.e. the TCN prediction and the indicator stack
       disagree, or don't agree strongly enough, or a required value is
       missing/NaN.

    ATR is intentionally NOT used in this decision (and no ATR-based
    SL/TP is computed here).

Edge-triggered signals
-----------------------
By default, the engine only emits BUY/SELL on the first candle a state
becomes true, then HOLD for every candle after that (even if the raw
condition is still true) until the opposite condition fires. This gives:

    BUY -> HOLD -> HOLD -> ... -> SELL -> HOLD -> ... -> BUY ...

instead of repeating BUY/SELL every candle the raw condition happens to
still hold. Set `SignalLogicConfig.edge_triggered = False` for the raw,
non-deduped behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# current_price is whatever you consider "now" -- typically the latest
# close. predicted_next_open comes straight out of the TCN (already
# inverse-scaled back to real price units).
#
# These are the core columns the module has always needed.
REQUIRED_COLUMNS = ("current_price", "ema9", "ema20", "predicted_next_open")

# Indicator columns used purely as CONFIRMATION filters on top of the TCN
# prediction. Mirrors INDICATOR_FEATURES from the feature-engineering side:
#
#   INDICATOR_FEATURES = {
#       "EMA9": ["ema9"], "EMA20": ["ema20"], "EMA50": ["ema50"],
#       "VWAP": ["vwap"], "RSI14": ["rsi14"],
#       "MACD": ["macd_line", "macd_signal", "macd_hist"],
#   }
#
# ema9/ema20 are already covered by REQUIRED_COLUMNS above.
CONFIRMATION_COLUMNS = (
    "ema50",
    "vwap",
    "rsi14",
    "macd_line",
    "macd_signal",
    "macd_hist",
)


@dataclass
class SignalLogicConfig:
    """
    Tunable parameters. Nothing here touches the model -- safe to change
    any time.
    """

    # Minimum predicted move, as a fraction of current_price, required to
    # act instead of HOLD. 0.0001 = 0.01% -- realistic for 1-minute bars
    # (0.1% on a ~$750 SPY bar would be a 75 cent move in one minute,
    # which basically never happens, so that threshold always HOLDs).
    threshold_pct: float = 0.0001

    # Optional minimum EMA9/EMA20 separation (as a fraction of price) to
    # avoid signaling on a razor-thin, noisy crossover. 0.0 disables this.
    # Only affects the informational `trend` column, not the BUY/SELL gate
    # (the BUY/SELL gate uses the full ema9/ema20/ema50 stack instead).
    min_ema_gap_pct: float = 0.0

    # If True (default), only emit BUY/SELL on the candle where the signal
    # state actually changes; every other candle is HOLD. If False, BUY/SELL
    # fires on every candle where the raw condition holds.
    edge_triggered: bool = True

    # RSI14 momentum band required to confirm a BUY. Deliberately NOT a
    # classic "oversold" threshold -- RSI14 here confirms bullish momentum
    # is present without being so high it's already an overbought extreme.
    rsi_buy_min: float = 50.0
    rsi_buy_max: float = 70.0

    # RSI14 momentum band required to confirm a SELL. Deliberately NOT a
    # classic "overbought" threshold -- RSI14 here confirms bearish
    # momentum is present without being so low it's already an
    # oversold extreme.
    rsi_sell_max: float = 50.0
    rsi_sell_min: float = 30.0


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Signal logic input is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )


def _missing_confirmation_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in CONFIRMATION_COLUMNS if c not in df.columns]


def compute_predicted_move_pct(df: pd.DataFrame) -> pd.Series:
    """
    (predicted_next_open - current_price) / current_price

    Positive => model expects price to rise.
    Negative => model expects price to fall.
    """
    return (df["predicted_next_open"] - df["current_price"]) / df["current_price"]


def trend_state(df: pd.DataFrame, min_gap_pct: float = 0.0) -> pd.Series:
    """
    "UP"   when ema9 is above ema20 by at least min_gap_pct
    "DOWN" when ema9 is below ema20 by at least min_gap_pct
    "FLAT" otherwise

    NOTE: this remains a lightweight, informational column (used for
    charting/back-compat). It is NOT the full confirmation gate used to
    decide BUY/SELL -- see `_bullish_confirmation` / `_bearish_confirmation`.
    """
    gap_pct = (df["ema9"] - df["ema20"]) / df["current_price"]

    state = pd.Series("FLAT", index=df.index)
    state[gap_pct > min_gap_pct] = "UP"
    state[gap_pct < -min_gap_pct] = "DOWN"
    return state


# ---------------------------------------------------------------------------
# Indicator confirmation filters
#
# Each function returns a boolean Series, aligned to df.index. Any row with
# a missing/NaN input naturally evaluates to False (pandas comparisons
# against NaN are False), which folds straight into HOLD downstream --
# no separate NaN branching required, and nothing crashes.
# ---------------------------------------------------------------------------

def _bullish_confirmation(df: pd.DataFrame, config: SignalLogicConfig) -> pd.Series:
    """
    All of:
        ema9 > ema20 > ema50
        current_price > vwap
        macd_line > macd_signal AND macd_hist > 0
        rsi_buy_min < rsi14 < rsi_buy_max
    """
    ema_stack = (df["ema9"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    above_vwap = df["current_price"] > df["vwap"]
    macd_bullish = (df["macd_line"] > df["macd_signal"]) & (df["macd_hist"] > 0)
    rsi_bullish = (df["rsi14"] > config.rsi_buy_min) & (df["rsi14"] < config.rsi_buy_max)

    confirmed = ema_stack & above_vwap & macd_bullish & rsi_bullish
    return confirmed.fillna(False)


def _bearish_confirmation(df: pd.DataFrame, config: SignalLogicConfig) -> pd.Series:
    """
    All of:
        ema9 < ema20 < ema50
        current_price < vwap
        macd_line < macd_signal AND macd_hist < 0
        rsi_sell_min < rsi14 < rsi_sell_max
    """
    ema_stack = (df["ema9"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    below_vwap = df["current_price"] < df["vwap"]
    macd_bearish = (df["macd_line"] < df["macd_signal"]) & (df["macd_hist"] < 0)
    rsi_bearish = (df["rsi14"] < config.rsi_sell_max) & (df["rsi14"] > config.rsi_sell_min)

    confirmed = ema_stack & below_vwap & macd_bearish & rsi_bearish
    return confirmed.fillna(False)


# ---------------------------------------------------------------------------
# Edge-triggering: collapse repeated BUY/BUY/BUY... or SELL/SELL/SELL... runs
# down to a single signal on the first candle of the run. A HOLD candle does
# NOT reset the position state -- only an opposite BUY/SELL flips it.
# ---------------------------------------------------------------------------

def _dedupe_signal_transitions(raw_signals: pd.Series) -> pd.Series:
    """
    Example:
        raw:    HOLD BUY BUY BUY HOLD SELL SELL BUY
        result: HOLD BUY HOLD HOLD HOLD SELL HOLD BUY
    """
    out = []
    last_actionable: Optional[str] = None

    for value in raw_signals:
        if value == Signal.HOLD.value:
            out.append(Signal.HOLD.value)
        elif value != last_actionable:
            out.append(value)
            last_actionable = value
        else:
            out.append(Signal.HOLD.value)

    return pd.Series(out, index=raw_signals.index)


# ---------------------------------------------------------------------------
# Strategy: TCN predicted move (primary) confirmed by the full indicator
# stack -- EMA9/20/50, VWAP, RSI14, MACD (secondary / confirmation only).
# ATR is intentionally not consulted here.
# ---------------------------------------------------------------------------

def tcn_prediction_strategy(df: pd.DataFrame, config: SignalLogicConfig) -> pd.Series:
    """
    Raw condition (before edge-triggering):
        BUY  when predicted_move_pct >  threshold_pct
             AND the full bullish indicator stack confirms
             (ema9>ema20>ema50, price>vwap, MACD bullish, RSI in bullish band)
        SELL when predicted_move_pct < -threshold_pct
             AND the full bearish indicator stack confirms
             (ema9<ema20<ema50, price<vwap, MACD bearish, RSI in bearish band)
        HOLD otherwise -- including whenever a required indicator value is
             missing/NaN, since confirmation simply fails closed to False.

    If config.edge_triggered is True (default), repeated BUY/SELL candles
    are collapsed so only the first candle of each new state fires.
    """
    _validate_columns(df)

    missing_confirmation = _missing_confirmation_columns(df)
    if missing_confirmation:
        # Indicator columns aren't present at all (as opposed to merely
        # containing NaN in some rows) -- there's nothing to confirm
        # against, so fail safe to HOLD for every row rather than crashing
        # or silently ignoring the confirmation requirement.
        return pd.Series(Signal.HOLD.value, index=df.index)

    move_pct = compute_predicted_move_pct(df)

    bullish_move = move_pct > config.threshold_pct
    bearish_move = move_pct < -config.threshold_pct

    confirm_bull = _bullish_confirmation(df, config)
    confirm_bear = _bearish_confirmation(df, config)

    raw_signals = pd.Series(Signal.HOLD.value, index=df.index)
    raw_signals[bullish_move & confirm_bull] = Signal.BUY.value
    raw_signals[bearish_move & confirm_bear] = Signal.SELL.value

    if config.edge_triggered:
        return _dedupe_signal_transitions(raw_signals)
    return raw_signals


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Thin wrapper so the strategy function can be swapped later (e.g. an
    RSI/MACD confirmation layer on top) without changing caller code.

    Usage:
        engine = SignalEngine()
        result_df = engine.generate(df)  # needs current_price/ema9/ema20/predicted_next_open
                                          # plus ema50/vwap/rsi14/macd_line/macd_signal/macd_hist
                                          # for indicator confirmation
    """

    def __init__(
        self,
        strategy=tcn_prediction_strategy,
        config: Optional[SignalLogicConfig] = None,
    ):
        self.strategy = strategy
        self.config = config or SignalLogicConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with new columns:
            - 'predicted_move_pct': the raw move % used in the decision
            - 'trend'             : "UP" / "DOWN" / "FLAT" (informational,
                                     ema9 vs ema20 only -- see trend_state)
            - 'signal'            : "BUY" / "SELL" / "HOLD"
        """
        _validate_columns(df)
        out = df.copy()
        out["predicted_move_pct"] = compute_predicted_move_pct(out)
        out["trend"] = trend_state(out, self.config.min_ema_gap_pct)
        out["signal"] = self.strategy(out, self.config)
        return out


# ---------------------------------------------------------------------------
# Convert signals into Lightweight Charts markers
# ---------------------------------------------------------------------------

def to_lightweight_charts_markers(df: pd.DataFrame, time_column: str = "time") -> list[dict]:
    """
    Convert a DataFrame that already has a 'signal' column (from
    SignalEngine.generate) into the marker format Lightweight Charts expects
    for `series.setMarkers([...])`.

    Only BUY and SELL rows produce markers; HOLD rows are skipped.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must have a 'signal' column. Run SignalEngine.generate() first.")
    if time_column not in df.columns:
        raise ValueError(f"DataFrame must have a '{time_column}' column for chart timestamps.")

    markers = []
    for _, row in df.iterrows():
        if row["signal"] == Signal.BUY.value:
            markers.append(
                {
                    "time": row[time_column],
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "BUY",
                }
            )
        elif row["signal"] == Signal.SELL.value:
            markers.append(
                {
                    "time": row[time_column],
                    "position": "aboveBar",
                    "color": "#ef5350",
                    "shape": "arrowDown",
                    "text": "SELL",
                }
            )
    return markers


if __name__ == "__main__":
    # Minimal smoke test using a tiny, clearly-labeled synthetic frame.
    # This block only sanity-checks the logic in isolation -- never used
    # by the real pipeline.
    #
    # Candles 0-2: full bullish stack (ema9>ema20>ema50, price>vwap, MACD
    # bullish, RSI in the 50-70 confirmation band) and a bullish TCN
    # prediction -- with edge_triggered=True you should see exactly one
    # BUY (candle 0), then HOLD while the state persists, then a SELL once
    # everything flips bearish and the TCN prediction turns bearish too.
    sample = pd.DataFrame(
        {
            "time": range(8),
            "current_price":      [100, 101, 102, 103, 103, 102, 101, 100],
            "ema9":                [99, 100, 101, 102, 102, 101.5, 101, 100.5],
            "ema20":               [98.5, 99.5, 99.8, 100, 100.5, 101.2, 101.5, 101.8],
            "ema50":               [98.0, 98.2, 98.5, 98.8, 99.0, 100.5, 101.8, 102.2],
            "vwap":                [99.5, 100.2, 101.0, 101.8, 102.0, 102.5, 102.0, 101.0],
            "rsi14":               [55, 58, 60, 62, 61, 48, 35, 32],
            "macd_line":           [0.5, 0.6, 0.7, 0.8, 0.7, -0.1, -0.6, -0.8],
            "macd_signal":         [0.3, 0.4, 0.5, 0.6, 0.6, 0.1, -0.2, -0.5],
            "macd_hist":           [0.2, 0.2, 0.2, 0.2, 0.1, -0.2, -0.4, -0.3],
            "predicted_next_open": [100.3, 101.4, 102.6, 103.5, 103.2, 101.5, 100.2, 99.5],
        }
    )
    engine = SignalEngine()
    result = engine.generate(sample)
    cols = [
        "current_price", "ema9", "ema20", "ema50", "vwap", "rsi14",
        "macd_line", "macd_signal", "macd_hist", "predicted_next_open",
        "predicted_move_pct", "trend", "signal",
    ]
    print(result[cols])
    print(to_lightweight_charts_markers(result))
