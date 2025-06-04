# monitoring/monitoring.py

from prometheus_client import start_http_server, Gauge
import time
import random

# Definisci alcune metriche di esempio
BALANCE_GAUGE = Gauge("aure_balance_usdt", "Bilancio attuale in USDT")
POSITION_GAUGE = Gauge("aure_position_btc", "Posizione aperta in BTC")
LAST_BACKTEST_GAUGE = Gauge("aure_last_backtest_timestamp", "Timestamp (epoch) ultimo backtest completato")

def start_metrics_server(port: int = 8000):
    """
    Avvia il server Prometheus sulla porta specificata.
    """
    start_http_server(port)
    print(f"Prometheus metrics server avviato su /metrics (porta {port})")

    # Esempio di update random: nella pratica, leggi da un DB o file
    while True:
        # Qui devi sostituire con letture reali da file/DB
        balance = random.uniform(8000, 12000)
        position = random.uniform(0, 1)
        last_bt = int(time.time())

        BALANCE_GAUGE.set(balance)
        POSITION_GAUGE.set(position)
        LAST_BACKTEST_GAUGE.set(last_bt)

        time.sleep(15)

if __name__ == "__main__":
    start_metrics_server()
