
import subprocess
import time
import os
from datetime import datetime, timedelta

# Stato ultima esecuzione
last_run = {
    "sentiment": datetime.min,
    "regime": datetime.min,
    "rl": datetime.min,
    "automl": datetime.min,
    "backtest": datetime.min
}

# Helper per lanciare i moduli
def run(module):
    print(f"▶ Avvio modulo: {module}")
    try:
        subprocess.Popen(["python3", "-m", module])
    except Exception as e:
        print(f"❌ Errore in {module}: {e}")

def loop_intelligente():
    print("🧠 Loop Intelligente Aure attivo...")
    while True:
        now = datetime.utcnow()

        # Sentiment ogni 60 minuti
        if now - last_run["sentiment"] > timedelta(minutes=60):
            run("data_ingestion.sentiment_ingest")
            last_run["sentiment"] = now

        # Regime ogni giorno alle 00:05 UTC
        if now.hour == 0 and now.minute < 10 and now - last_run["regime"] > timedelta(hours=23):
            run("unsupervised.regime_detector")
            last_run["regime"] = now

        # RL training ogni notte alle 01:00 UTC
        if now.hour == 1 and now.minute < 10 and now - last_run["rl"] > timedelta(hours=23):
            run("rl.train_rl_agent")
            last_run["rl"] = now

        # AutoML tuning ogni lunedì
        if now.weekday() == 0 and now - last_run["automl"] > timedelta(days=6):
            run("automl.optuna_tune")
            last_run["automl"] = now

        # Backtest ogni giorno alle 23:00 UTC
        if now.hour == 23 and now.minute < 10 and now - last_run["backtest"] > timedelta(hours=23):
            run("backtester.advanced_backtester")
            last_run["backtest"] = now

        # Monitoring sempre attivo
        run("monitoring.monitoring")

        # Pausa 60s prima del prossimo ciclo
        time.sleep(60)

if __name__ == "__main__":
    try:
        loop_intelligente()
    except KeyboardInterrupt:
        print("🛑 Interrotto manualmente")
