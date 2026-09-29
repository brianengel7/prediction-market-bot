@echo off
cd /d C:\Users\surfe\Downloads\prediction-market-bot

if not exist logs mkdir logs

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" -m src.backtest.market_backfill >> logs\market_history_refresh.log 2>&1