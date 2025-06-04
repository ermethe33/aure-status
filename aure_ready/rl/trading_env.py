# rl/trading_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from storage.timeseries_storage import query_ohlcv
import ta

class TradingEnv(gym.Env):
    """
    Ambiente Gym per trading su un singolo asset (es. BTCUSDT).
    Osservazione: array di feature (prezzi, indicatori) su t e n step precedenti.
    Azione: 0 = hold, 1 = buy (long), 2 = sell (close long), (per semplicità non short).
    Reward: variazione di equity percentuale (o log-return).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, symbol: str, start: str, stop: str, window_size: int = 50, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()

        self.symbol = symbol
        self.start = pd.to_datetime(start)
        self.stop = pd.to_datetime(stop)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Carica i dati storici
        df = query_ohlcv(self.symbol, self.start.isoformat()+"Z", self.stop.isoformat()+"Z")
        if df.empty:
            raise ValueError(f"Nessun dato per {symbol} su {start}–{stop}")
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.raw_df = df.copy()

        # Calcolo di alcune feature tecniche
        self.df = self._add_indicators(df)
        self.n_steps = len(self.df) - self.window_size

        # Azione: Discrete(3): hold, buy, sell
        self.action_space = spaces.Discrete(3)

        # Osservazione: window_size barre × (close, ema-x, rsi, ecc). Esempio: 5 features
        n_features = self.df.shape[1] - 1  # escludo timestamp
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, n_features), dtype=np.float32)

        # Vars interne
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # quantità di asset in portafoglio
        self.entry_price = 0.0
        self.total_equity = initial_balance

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["ema_10"] = ta.trend.ema_indicator(df["close"], window=10)
        df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["volatility"] = df["close"].rolling(window=20).std()
        df = df.dropna().reset_index(drop=True)
        return df

    def reset(self):
        """
        Reimposta l'ambiente e ritorna la prima osservazione.
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_equity = self.initial_balance

        return self._get_observation()

    def _get_observation(self):
        """
        Ritorna le ultime window_size barre (features) come osservazione.
        """
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step].copy()
        obs = frame.drop(columns=["timestamp"]).values  # shape=(window_size, n_features)
        return obs.astype(np.float32)

    def step(self, action: int):
        """
        Esegue l'azione: 0=hold, 1=buy, 2=sell. Calcola reward come variazione di equity.
        Ritorna (obs_next, reward, done, info).
        """
        done = False
        info = {}

        price = self.df.iloc[self.current_step]["close"]
        timestamp = self.df.iloc[self.current_step]["timestamp"]

        # Esegui azione
        if action == 1 and self.position == 0:
            # apri posizione long con tutto il capitale disponibile
            self.position = self.balance / price
            self.entry_price = price
            self.balance = 0.0

        elif action == 2 and self.position > 0:
            # chiudi posizione
            proceeds = self.position * price
            self.balance = proceeds
            self.position = 0.0
            self.entry_price = 0.0

        # Calcola nuova equity
        self.total_equity = self.balance + self.position * price

        # Reward come variazione percentuale di equity rispetto al passo precedente
        prev_price = self.df.iloc[self.current_step - 1]["close"]
        prev_equity = self.balance + self.position * prev_price
        reward = (self.total_equity - prev_equity) / (prev_equity + 1e-9)

        # Avanza passo
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, info

    def render(self, mode="human"):
        """
        (opzionale) Implementa visualizzazione dello stato corrente.
        """
        price = self.df.iloc[self.current_step]["close"]
        print(f"Step: {self.current_step}, Prezzo: {price:.2f}, Balance: {self.balance:.2f}, Position: {self.position:.6f}, Equity: {self.total_equity:.2f}")

    def close(self):
        pass
