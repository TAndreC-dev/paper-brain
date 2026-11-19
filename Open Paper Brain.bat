@echo off
rem Double-click to open Paper Brain in your browser.
cd /d "%~dp0"

rem Already running? Then just open the tab.
netstat -ano | findstr /r /c:":8501 .*LISTENING" >nul
if not errorlevel 1 (
    start "" "http://localhost:8501"
    exit /b
)

rem Starts the app in a minimised window; it opens the browser tab itself.
rem Closing that window stops Paper Brain.
start "Paper Brain (close this window to stop)" /min ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
