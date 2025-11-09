from data_fetcher import fetch_upcoming_props_with_stats
from model import train_model, predict_props
from utils import filter_top_props, send_webhook
import os
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
TOP_N = int(os.getenv("TOP_N", 5))

def main():
    print("Fetching player stats, odds, and injuries...")
    df = fetch_upcoming_props_with_stats()

    print("Training model...")
    model, scaler = train_model(df)

    print("Predicting Over/Under confidence...")
    df = predict_props(df, model, scaler)

    filtered, top_props = filter_top_props(df, CONFIDENCE_THRESHOLD, TOP_N)

    print("Filtered props ≥ threshold:")
    print(filtered)

    print(f"Top {TOP_N} props:")
    print(top_props)

    print("Sending top props to webhook...")
    send_webhook(top_props)

if __name__ == "__main__":
    main()
