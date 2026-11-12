@echo off
title Running Automated Test Suite
echo =====================================================================
echo   Running Pytest Automated Unit and Integration Test Suite
echo =====================================================================
echo.
python -m pytest -v
echo.
echo =====================================================================
pause

