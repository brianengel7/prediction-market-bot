@echo off
set PYTHONUTF8=1
cd /d C:\Users\surfe\Downloads\prediction-market-bot

if not exist logs mkdir logs

echo ============================================================ >> logs\shadow_trader.log
echo NBM STRATEGY >> logs\shadow_trader.log
echo ============================================================ >> logs\shadow_trader.log

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" -m src.kalshi.v2_shadow_trader --save >> logs\shadow_trader.log 2>&1

echo ============================================================ >> logs\shadow_trader.log
echo MARKET-ONLY STRATEGY >> logs\shadow_trader.log
echo ============================================================ >> logs\shadow_trader.log

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" -m src.kalshi.v2_market_only_shadow_trader --save --live >> logs\shadow_trader.log 2>&1