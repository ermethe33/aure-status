# automl/optuna_tune.py

import os
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from storage.timeseries_storage import query_ohlcv
from models.supervised_model import prepare_dataset
from dotenv import load_dotenv

load_dotenv()

SYMBOL = "BTCUSDT"
STUDY_NAME = f"optuna_xgb_{SYMBOL}"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
MODEL_DIR = "models/artifacts"

def objective(trial):
    # Parametri da ottimizzare
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True)
    }

    # Preparo dati (ultimi 60 giorni)
    today = datetime.utcnow()
    start = (today - timedelta(days=60)).isoformat() + "Z"
    stop = today.isoformat() + "Z"
    df = query_ohlcv(SYMBOL, start, stop)
    dataset = prepare_dataset(df)
    X = dataset[["rsi", "ema_10", "ema_50", "volatility"]]
    y = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(param, dtrain, num_boost_round=100, evals=[(dtest, "eval")],
                    early_stopping_rounds=10, verbose_eval=False)
    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    return acc  # ottimizziamo accuracy

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, storage=STORAGE_NAME, load_if_exists=True)
    study.optimize(objective, n_trials=50)  # esegui 50 trial

    print("Migliori parametri:", study.best_params)
    print("Best accuracy:", study.best_value)

    # Salvo parametri e posso addestrare un modello definitivo con questi hyperparam
    best_param = study.best_params
    best_param.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    today = datetime.utcnow()
    start = (today - timedelta(days=60)).isoformat() + "Z"
    stop = today.isoformat() + "Z"
    df = query_ohlcv(SYMBOL, start, stop)
    dataset = prepare_dataset(df)
    X = dataset[["rsi", "ema_10", "ema_50", "volatility"]]
    y = dataset["target"]
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(best_param, dtrain, num_boost_round=100)
    timestamp = today.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"xgb_best_{SYMBOL}_{timestamp}.bin")
    model.save_model(model_path)
    print(f"Modello ottimizzato salvato in {model_path}")

if __name__ == "__main__":
    main()
