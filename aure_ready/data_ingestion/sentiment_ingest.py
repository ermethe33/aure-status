# data_ingestion/sentiment_ingest.py

import os
from datetime import datetime, timedelta
import tweepy
import pandas as pd
from textblob import TextBlob
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

load_dotenv()

# Credenziali Twitter (X)
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

QUERY = "BTC OR #Bitcoin -is:retweet lang:en"
MAX_TWEETS = 100  # numero di tweet da recuperare

def authenticate_twitter():
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    return client

def fetch_recent_tweets(client, query: str, max_results: int = MAX_TWEETS):
    """
    Recupera tweet dalle ultime 1–2 ore (dipende dai rate limits).
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    tweets = client.search_recent_tweets(query=query, start_time=start_time.isoformat()+"Z",
                                         end_time=end_time.isoformat()+"Z", max_results=100)
    return tweets.data if tweets and tweets.data else []

def compute_sentiment(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity

def main():
    # 1) Autenticazione
    client = authenticate_twitter()

    # 2) Scarica tweet recenti
    tweets = fetch_recent_tweets(client, QUERY, MAX_TWEETS)
    if not tweets:
        print("Nessun tweet recuperato.")
        return

    # 3) Calcola sentiment medio
    sentiments = []
    for tw in tweets:
        score = compute_sentiment(tw.text)
        sentiments.append(score)
    avg_sentiment = sum(sentiments) / (len(sentiments) + 1e-9)

    # 4) Invia in InfluxDB come punto singolo
    influx = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    now = datetime.utcnow()

    point = (
        Point("BTCUSDT_sentiment")
        .tag("source", "twitter")
        .field("avg_sentiment", float(avg_sentiment))
        .time(now, WritePrecision.NS)
    )
    write_api.write(bucket=INFLUXDB_BUCKET, record=point)
    print(f"[{now}] Sentiment medio Twitter inserito: {avg_sentiment:.4f}")

if __name__ == "__main__":
    main()
