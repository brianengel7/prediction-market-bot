@echo off
setlocal

cd /d C:\Users\surfe\Downloads\prediction-market-bot

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).AddDays(-3).ToString('yyyy-MM-dd')"') do set START_DATE=%%i

echo Running synchronized quote collector from %START_DATE%

C:\Users\surfe\Downloads\prediction-market-bot\.venv\Scripts\python.exe -m collect_synchronized_quotes --start-date %START_DATE%

endlocal