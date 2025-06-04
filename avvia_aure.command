#!/bin/bash
cd ~/Desktop/aure_operativo
source venv/bin/activate

# Avvio Logger
echo "📝 Avvio logger..."
python3 logger.py &

# Avvio Loop Intelligente
echo "🧠 Avvio loop_intelligente..."
python3 loop_intelligente.py &

# Avvio Loop di Fallback
echo "🛡 Avvio loop_auto_executor..."
python3 loop_auto_executor.py &

# (Opzionale) Avvio dashboard o notificatore
# echo "📊 Avvio dashboard..."
# python3 dashboard.py &

# echo "💬 Avvio notificatore WhatsApp..."
# python3 notificatore_whatsapp.py &

wait
