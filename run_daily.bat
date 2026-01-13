@echo off
echo ==========================================
echo      BDP PREMIER - DAILY UPDATE
echo ==========================================
echo.
echo 1. Fetching Latest Odds & Calculating EV...
cd /d "%~dp0nfl_ev_betting_engine"
python scripts/update_dashboard.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Update failed! Check internet connection or API keys.
    pause
    exit /b %errorlevel%
)

echo.
echo 2. Dashboard Updated Successfully!
echo    - NFL Odds: Refreshed
echo    - Predictions: Generated
echo    - Edges: Found
echo.
echo To view dashboard: http://localhost:8080/
echo.
pause
