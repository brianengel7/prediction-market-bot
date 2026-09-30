@echo off
set PYTHONUTF8=1
cd /d C:\Users\surfe\Downloads\prediction-market-bot

if not exist logs mkdir logs

"C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe" main.py >> logs\weather_model.log 2>&1