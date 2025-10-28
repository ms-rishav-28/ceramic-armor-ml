@echo off
REM ============================================================================
REM CERAMIC ARMOR ML PROJECT - WINDOWS SETUP SCRIPT
REM ============================================================================
REM This script sets up the complete Ceramic Armor ML project environment
REM on Windows 11 Pro 64-bit systems with automated dependency installation
REM and configuration file generation.
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo CERAMIC ARMOR ML PROJECT - WINDOWS SETUP
echo ============================================================================
echo.
echo This script will:
echo   1. Check system requirements
echo   2. Create/activate conda environment
echo   3. Install Python dependencies
echo   4. Create directory structure
echo   5. Generate configuration files
echo   6. Validate setup
echo.

REM Check if conda is installed
echo [1/6] Checking system requirements...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Check Python version requirement
echo   - Conda found: OK
python --version 2>nul | findstr "3.9 3.10 3.11 3.12" >nul
if %errorlevel% neq 0 (
    echo WARNING: Python 3.9-3.12 recommended for optimal compatibility
)

REM Check available disk space (require at least 5GB)
for /f "tokens=3" %%a in ('dir /-c %cd% 2^>nul ^| find "bytes free"') do set freespace=%%a
set /a freespace_gb=!freespace:~0,-9!
if !freespace_gb! lss 5 (
    echo WARNING: Less than 5GB free space available. Setup may fail.
    echo Available space: !freespace_gb!GB
)

echo   - System requirements: OK
echo.

REM Create conda environment
echo [2/6] Setting up conda environment...
set ENV_NAME=ceramic-armor-ml

REM Check if environment already exists
conda env list | findstr "%ENV_NAME%" >nul
if %errorlevel% equ 0 (
    echo   - Environment '%ENV_NAME%' already exists
    choice /c YN /m "Do you want to recreate it? (Y/N)"
    if !errorlevel! equ 1 (
        echo   - Removing existing environment...
        conda env remove -n %ENV_NAME% -y
        if !errorlevel! neq 0 (
            echo ERROR: Failed to remove existing environment
            pause
            exit /b 1
        )
    ) else (
        echo   - Using existing environment
        goto :activate_env
    )
)

echo   - Creating conda environment with Python 3.11...
conda create -n %ENV_NAME% python=3.11 -y
if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

:activate_env
echo   - Activating environment...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate conda environment
    echo Try running: conda activate %ENV_NAME%
    pause
    exit /b 1
)

echo   - Environment setup: OK
echo.

REM Install dependencies
echo [3/6] Installing Python dependencies...
echo   - Installing packages from requirements.txt...
echo   - This may take 10-15 minutes depending on your internet connection...

REM Install pip packages with progress indication
pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

REM Install requirements with verbose output for progress tracking
pip install -r requirements.txt --verbose
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    echo.
    echo Common solutions:
    echo   1. Check internet connection
    echo   2. Try: pip install --upgrade pip setuptools wheel
    echo   3. Install Visual Studio Build Tools if needed
    echo   4. Run as Administrator if permission errors occur
    pause
    exit /b 1
)

echo   - Python dependencies: OK
echo.

REM Create directory structure
echo [4/6] Creating directory structure...

REM Directories that should exist (verify and create if missing)
set DIRS=data\raw data\processed data\features data\splits data\data_collection results\models results\predictions results\metrics results\figures results\reports logs config notebooks

for %%d in (%DIRS%) do (
    if not exist "%%d" (
        echo   - Creating directory: %%d
        mkdir "%%d" 2>nul
        if !errorlevel! neq 0 (
            echo WARNING: Failed to create directory: %%d
        )
    ) else (
        echo   - Directory exists: %%d
    )
)

echo   - Directory structure: OK
echo.

REM Generate configuration files
echo [5/6] Generating configuration files...

REM Create api_keys.yaml if it doesn't exist
if not exist "config\api_keys.yaml" (
    echo   - Creating config\api_keys.yaml from template...
    copy "config\api_keys.yaml.example" "config\api_keys.yaml" >nul
    if !errorlevel! neq 0 (
        echo WARNING: Failed to create api_keys.yaml
    ) else (
        echo   - IMPORTANT: Edit config\api_keys.yaml with your actual API keys
    )
) else (
    echo   - config\api_keys.yaml already exists
)

REM Create .env file for environment variables
if not exist ".env" (
    echo   - Creating .env file...
    (
        echo # Environment variables for Ceramic Armor ML Project
        echo # Intel CPU Optimization
        echo OMP_NUM_THREADS=20
        echo MKL_NUM_THREADS=20
        echo NUMEXPR_NUM_THREADS=20
        echo OPENBLAS_NUM_THREADS=20
        echo.
        echo # Project paths
        echo PROJECT_ROOT=%cd%
        echo DATA_PATH=%cd%\data
        echo RESULTS_PATH=%cd%\results
        echo CONFIG_PATH=%cd%\config
        echo.
        echo # API Configuration
        echo MP_API_KEY=YOUR_MATERIALS_PROJECT_API_KEY_HERE
    ) > .env
    echo   - Created .env file
) else (
    echo   - .env file already exists
)

REM Create Windows-specific batch files for common operations
echo   - Creating Windows utility scripts...

REM Create activate environment script
(
    echo @echo off
    echo REM Activate the ceramic-armor-ml conda environment
    echo call conda activate ceramic-armor-ml
    echo if %%errorlevel%% neq 0 (
    echo     echo ERROR: Failed to activate environment
    echo     pause
    echo     exit /b 1
    echo ^)
    echo echo Environment activated successfully
    echo echo You can now run Python scripts in this environment
    echo cmd /k
) > activate_env.bat

REM Create quick test script
(
    echo @echo off
    echo REM Quick test of the ceramic armor ML setup
    echo call conda activate ceramic-armor-ml
    echo if %%errorlevel%% neq 0 (
    echo     echo ERROR: Failed to activate environment
    echo     pause
    echo     exit /b 1
    echo ^)
    echo echo Running setup validation...
    echo python scripts\00_validate_setup.py
    echo pause
) > quick_test.bat

REM Create run minimal test script
(
    echo @echo off
    echo REM Run minimal test pipeline
    echo call conda activate ceramic-armor-ml
    echo if %%errorlevel%% neq 0 (
    echo     echo ERROR: Failed to activate environment
    echo     pause
    echo     exit /b 1
    echo ^)
    echo echo Running minimal test pipeline...
    echo python scripts\run_minimal_test.py
    echo pause
) > run_minimal_test.bat

echo   - Configuration files: OK
echo.

REM Validate setup
echo [6/6] Validating setup...
echo   - Testing Python imports...

REM Test critical imports
python -c "import numpy, pandas, sklearn, xgboost, catboost, shap, pymatgen; print('Core imports: OK')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Some core imports failed. Check installation.
) else (
    echo   - Core imports: OK
)

REM Test Intel optimizations
python -c "from sklearnex import patch_sklearn; patch_sklearn(); print('Intel optimizations: OK')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Intel optimizations not available
) else (
    echo   - Intel optimizations: OK
)

REM Test project imports
python -c "import sys; sys.path.append('src'); import utils, data_collection; print('Project imports: OK')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Project imports failed. Check src/__init__.py files
) else (
    echo   - Project imports: OK
)

echo.
echo ============================================================================
echo SETUP COMPLETE!
echo ============================================================================
echo.
echo Next steps:
echo   1. Edit config\api_keys.yaml with your Materials Project API key
echo   2. Get your API key at: https://materialsproject.org/api
echo   3. Run quick_test.bat to validate your setup
echo   4. Run run_minimal_test.bat to test the pipeline
echo.
echo Useful commands:
echo   - activate_env.bat          : Activate the conda environment
echo   - quick_test.bat            : Run setup validation
echo   - run_minimal_test.bat      : Run minimal test pipeline
echo.
echo Environment name: %ENV_NAME%
echo To manually activate: conda activate %ENV_NAME%
echo.
echo For troubleshooting, see the Windows documentation in docs\
echo.

REM Create setup completion marker
echo Setup completed on %date% %time% > setup_complete.txt
echo Environment: %ENV_NAME% >> setup_complete.txt
echo Python version: >> setup_complete.txt
python --version >> setup_complete.txt 2>&1

echo Press any key to exit...
pause >nul

endlocal