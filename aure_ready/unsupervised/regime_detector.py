# unsupervised/regime_detector.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from storage.timeseries_storage import query_ohlcv
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "BTCUSDT"
WINDOW_SIZE = 60  # minuti per aggregazione
N_CLUSTERS = 3    # es. 3 regimi di mercato

def extract_rolling_features(df: pd.DataFrame, window: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    Per ogni istante t, calcola feature su [t-window : t]:
    - rendimento medio (mean return)
    - volatilità (std return)
    - skew, kurtosis
    - momentum (ultimo ritorno)
    Ritorna un df con index = timestamp di fine finestra.
    """
    df = df.copy().set_index("timestamp")
    df["return"] = df["close"].pct_change().fillna(0)
    features = []
    timestamps = []

    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        ts = df.index[i]
        mean_ret = window_df["return"].mean()
        vol = window_df["return"].std()
        skew = window_df["return"].skew()
        kurt = window_df["return"].kurtosis()
        momentum = window_df["return"].iloc[-1]
        features.append([mean_ret, vol, skew, kurt, momentum])
        timestamps.append(ts)

    feat_df = pd.DataFrame(features, columns=["mean_ret", "vol", "skew", "kurt", "momentum"], index=timestamps)
    return feat_df.dropna()

def detect_regimes(symbol: str, start: str, stop: str):
    df = query_ohlcv(symbol, start, stop)
    if df.empty:
        print("Nessun dato per regime detection.")
        return

    feat_df = extract_rolling_features(df, window=WINDOW_SIZE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    regimes = kmeans.fit_predict(X_scaled)
    feat_df["regime"] = regimes

    # Plot regimi nel tempo (in background, su equity o su prezzo)
    plt.figure(figsize=(12, 4))
    plt.scatter(feat_df.index, df.set_index("timestamp").loc[feat_df.index]["close"], c=feat_df["regime"], cmap="tab10", s=5)
    plt.title(f"Regime di mercato ({N_CLUSTERS} cluster) su {symbol}")
    plt.xlabel("Tempo")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.show()

    # Salvo CSV con regimi
    out_csv = f"regimes_{symbol}_{start[:10]}_{stop[:10]}.csv"
    feat_df.to_csv(out_csv)
    print(f"Regimi salvati in {out_csv}")

if __name__ == "__main__":
    today = datetime.utcnow()
    month_ago = today - timedelta(days=30)
    detect_regimes(SYMBOL, month_ago.isoformat()+"Z", today.isoformat()+"Z")
