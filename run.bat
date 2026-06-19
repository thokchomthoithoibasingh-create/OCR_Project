@echo off
REM run.bat
REM ---------
REM Convenience launcher for Windows. Double-click this file (or run it
REM from CMD) from the project root to activate the virtual environment
REM and start the OCR CLI app, without typing the activation command
REM every time.

echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Could not find venv\Scripts\activate.bat
    echo Make sure you created the virtual environment first:
    echo     python -m venv venv
    pause
    exit /b 1
)

echo Starting OCR CLI application...
python src\main.py

pause
