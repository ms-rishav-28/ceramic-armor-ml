@echo off
echo 🚀 Installing Ceramic Armor ML Pipeline Dependencies
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist requirements.txt (
    echo ❌ Error: requirements.txt not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo 🔄 Installing all dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Error: Failed to install requirements
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo 📋 Ready to run tests with: python -m pytest tests/ -v
echo.
pause