@echo off
title Launching Adaptive Cloud Scheduler Dashboard
echo =====================================================================
echo   Behavior-Aware Adaptive Cloud Resource Scheduling Dashboard
echo =====================================================================
echo.
echo Launching Streamlit web application...
echo The dashboard will automatically open in your default browser.
echo (Press Ctrl+C in this window to stop the server)
echo.
python -m streamlit run app.py
pause

