import os
import json
import time
from datetime import datetime
from api_bridge import get_account_balance, get_sniper_status, get_grid_bots
from dotenv import load_dotenv

load_dotenv(dotenv_path="api_config.env")

REPO_PATH = os.path.expanduser("~/Desktop/aure-status")
STATUS_PATH = os.path.join(REPO_PATH, "status.json")

def aggiorna_status():
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Dati simulati, sostituire con chiamate API reali
    binance = {
        "USDT": get_account_balance("USDT", exchange="binance"),
        "ADA": get_account_balance("ADA", exchange="binance")
    }

    pionex = {
        "USDT": get_account_balance("USDT", exchange="pionex"),
        "WIF": get_account_balance("WIF", exchange="pionex")
    }

    stato = {
        "timestamp": now,
        "status": "attivo",
        "exchange": "Pionex",
        "bot_attivi": len(get_grid_bots()),
        "sniper_attivi": len(get_sniper_status()),
        "wallet": {
            "binance": binance,
            "pionex": pionex
        }
    }

    with open(STATUS_PATH, "w") as f:
        json.dump(stato, f, indent=2)
    print("✅ status.json aggiornato")

    os.chdir(REPO_PATH)
    os.system("git add status.json")
    os.system(f'git commit -m "Aggiornamento automatico {now}"')
    os.system("git push")
    print("✅ Commit e push completati")

if __name__ == "__main__":
    while True:
        aggiorna_status()
        time.sleep(600)  # ogni 10 minuti
