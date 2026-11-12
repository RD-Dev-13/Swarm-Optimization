@echo off
title Running Multi-Scenario Experiments
echo =====================================================================
echo   Running Comprehensive Experimental Benchmark Suite
echo =====================================================================
echo.
python -m experiments.run_experiments
echo.
echo =====================================================================
echo Results exported to results/experiments/ and results/graphs/
echo =====================================================================
pause

