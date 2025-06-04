# backtester/advanced_backtester.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from storage.timeseries_storage import query_ohlcv
import ta
import logging
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "BTCUSDT"
COMMISSION_PCT = 0.001  # 0.1% commissione a ogni trade
SLIPPAGE_PCT = 0.0005  # 0.05% slippage per round-trip
DEFAULT_ORDER_TYPE = "market"  # 'market' o 'limit'
LOG = logging.getLogger("advanced_backtester")

def apply_slippage(price: float, side: str) -> float:
    """
    Applica slippage al prezzo:
    - se side == 'buy', il bot paga un prezzo maggiore (price * (1 + slippage))
    - se side == 'sell', il bot vende a un prezzo minore (price * (1 - slippage))
    """
    if side == "buy":
        return price * (1 + SLIPPAGE_PCT)
    elif side == "sell":
        return price * (1 - SLIPPAGE_PCT)
    else:
        return price

def compute_commission(value: float) -> float:
    """
    Calcola la commissione totale (commessione % basata sul trade value).
    """
    return abs(value) * COMMISSION_PCT

def load_data(symbol: str, start: str, stop: str) -> pd.DataFrame:
    df = query_ohlcv(symbol, start, stop)
    if df.empty:
        LOG.warning("Nessun dato disponibile per periodo %s – %s", start, stop)
    return df.copy()

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Esempio di strategia: crossover EMA(10,50).
    Ritorna df con colonne: ['timestamp', 'signal'] dove signal = 1(long), -1(short), 0(flat).
    """
    df = df.copy()
    df["ema_short"] = ta.trend.ema_indicator(df["close"], window=10)
    df["ema_long"]  = ta.trend.ema_indicator(df["close"], window=50)
    df["signal_raw"] = 0
    df.loc[df["ema_short"] > df["ema_long"], "signal_raw"] = 1
    df.loc[df["ema_short"] < df["ema_long"], "signal_raw"] = -1
    df["signal"] = df["signal_raw"].shift(1).fillna(0).astype(int)
    return df[["timestamp", "signal"]]

def run_advanced_backtest(symbol: str, start: str, stop: str, initial_capital: float = 10000.0, order_type: str = DEFAULT_ORDER_TYPE):
    """
    Esegue backtest sul periodo [start, stop], partendo da initial_capital (in USDT).
    order_type: 'market' o 'limit' (limit simula un price fittizio equal to close).
    """
    df = load_data(symbol, start, stop)
    if df.empty:
        print("Nessun dato per backtest avanzato.")
        return

    # Genera segnali
    signals = generate_signals(df)
    df = df.merge(signals, on="timestamp", how="left").fillna(0)

    # Prepara colonne per equity, posizioni, PnL
    df["position"] = 0  # 1 = long pieno, -1 = short pieno, 0 = flat
    df["trade_price"] = np.nan
    df["cash"] = initial_capital
    df["holding"] = 0.0  # quantità di asset
    df["total_equity"] = initial_capital

    pos = 0  # posizione in unità base (es. BTC)
    cash = initial_capital
    last_signal = 0

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]
        timestamp = row["timestamp"]
        signal = row["signal"]
        price = row["close"]

        # Se il segnale cambia (da flat a long o da long a flat, ecc.)
        if signal != last_signal:
            # Decide se deve aprire/chiudere posizione
            if last_signal == 0 and signal == 1:
                # apri long
                trade_price = apply_slippage(price, side="buy") if order_type == "market" else price
                units = cash / trade_price  # investo tutto in long
                cash_spent = units * trade_price
                commission = compute_commission(cash_spent)
                cash_after = cash - cash_spent - commission
                pos = units
                cash = cash_after
                df.at[idx, "trade_price"] = trade_price
                LOG.info("[%s] BUY %.6f @ %.2f (commissione=%.2f)", timestamp, units, trade_price, commission)

            elif last_signal == 1 and signal == 0:
                # chiudi long
                trade_price = apply_slippage(price, side="sell") if order_type == "market" else price
                proceeds = pos * trade_price
                commission = compute_commission(proceeds)
                cash_after = cash + proceeds - commission
                pnl = proceeds - commission - (pos * df.iloc[idx-1]["close"])  # approssimazione cost-basis
                pos = 0
                cash = cash_after
                df.at[idx, "trade_price"] = trade_price
                LOG.info("[%s] SELL %.6f @ %.2f (commissione=%.2f) PnL=%.2f", timestamp, df.iloc[idx-1]["holding"], trade_price, commission, pnl)

            # (puoi aggiungere short in analogo modo se implementi)
            last_signal = signal

        # Aggiorna stato riga
        df.at[idx, "position"] = last_signal
        df.at[idx, "holding"] = pos
        df.at[idx, "cash"] = cash
        df.at[idx, "total_equity"] = cash + pos * price

    # Calcola metriche finali
    equity_series = df["total_equity"]
    total_return = equity_series.iloc[-1] / initial_capital - 1
    max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
    strat_returns = equity_series.pct_change().fillna(0)
    sharpe = np.sqrt(252 * 24 * 60) * (strat_returns.mean() / (strat_returns.std() + 1e-9))

    print(f"--- BACKTEST AVANZATO {symbol} ---")
    print(f"Periodo: {start} – {stop}")
    print(f"Return totale: {total_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio (annualizzato): {sharpe:.2f}")

    # Salvo CSV e grafico equity
    out_csv = f"advanced_backtest_{symbol}_{start[:10]}_{stop[:10]}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Risultati salvati in {out_csv}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        df.set_index("timestamp")["total_equity"].plot(title=f"Equity Curve Avanzata {symbol}")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    today = datetime.utcnow()
    week_ago = today - timedelta(days=7)
    run_advanced_backtest(SYMBOL, week_ago.isoformat()+"Z", today.isoformat()+"Z", initial_capital=10000.0, order_type="market")
