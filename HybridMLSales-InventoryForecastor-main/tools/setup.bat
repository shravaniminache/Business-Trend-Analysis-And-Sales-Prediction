@echo off
REM ==============================================================================
REM Manufacturing Dashboard - Automated Setup Script
REM For Windows
REM ==============================================================================

echo ==========================================
echo Manufacturing Dashboard Setup
echo ==========================================
echo.

REM Check Python installation
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python found
    python --version
) else (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo.

REM Check pip installation
echo Step 2: Checking pip installation...
pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] pip found
) else (
    echo [ERROR] pip not found!
    echo Installing pip...
    python -m ensurepip --upgrade
)

echo.

REM Install dependencies
echo Step 3: Installing required packages...
echo This may take 2-3 minutes...
echo.

pip install streamlit pandas numpy plotly

if %errorlevel% equ 0 (
    echo.
    echo [OK] All packages installed successfully!
) else (
    echo.
    echo [ERROR] Installation failed. Trying alternative method...
    python -m pip install streamlit pandas numpy plotly
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run the dashboard, execute:
echo.
echo     streamlit run manufacturing_dashboard.py
echo.
echo Or if that doesn't work:
echo.
echo     python -m streamlit run manufacturing_dashboard.py
echo.
echo The dashboard will open automatically in your browser!
echo ==========================================
echo.
pause
