@echo off
set PYTHONUTF8=1

cd /d C:\Users\surfe\Downloads\prediction-market-bot

if not exist logs mkdir logs

echo ============================================================ >> logs\shadow_settlement.log
echo MARKET HISTORY REFRESH >> logs\shadow_settlement.log
echo ============================================================ >> logs\shadow_settlement.log

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" -m src.backtest.market_backfill >> logs\shadow_settlement.log 2>&1

echo. >> logs\shadow_settlement.log
echo ============================================================ >> logs\shadow_settlement.log
echo SHADOW TRADE SETTLEMENT >> logs\shadow_settlement.log
echo ============================================================ >> logs\shadow_settlement.log

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" -m src.kalshi.v2_shadow_settler --save >> logs\shadow_settlement.log 2>&1

echo. >> logs\shadow_settlement.log