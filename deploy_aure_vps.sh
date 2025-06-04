#!/bin/bash

echo "=== 🚀 Inizio installazione Aure su VPS ==="

# Aggiorna pacchetti
sudo apt update && sudo apt upgrade -y

# Installa dipendenze di sistema
sudo apt install -y python3 python3-pip python3-venv git unzip tmux

# Crea cartella progetto
mkdir -p ~/aure_operativo
cd ~/aure_operativo

# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa requirements base
echo "📦 Installazione librerie Python..."
pip install --upgrade pip
pip install python-dotenv pandas numpy requests ccxt ta             scikit-learn xgboost joblib schedule flask             influxdb-client prometheus-client elasticsearch             gunicorn psutil

# Clona progetto da GitHub se disponibile
# git clone https://github.com/tuo_repo/aure_status.git .

# Crea cartella logs e data
mkdir -p logs data

# File di test rapido
echo 'print("✅ Aure pronto!")' > test.py

echo "=== ✅ Installazione completata! ==="
echo "Puoi ora eseguire:"
echo "cd ~/aure_operativo"
echo "source venv/bin/activate"
echo "python3 test.py"
