# ============================================================================
# CERAMIC ARMOR ML PROJECT - WINDOWS POWERSHELL SETUP SCRIPT
# ============================================================================
# PowerShell version of the setup script for advanced users
# Requires PowerShell 5.1 or later
# ============================================================================

param(
    [switch]$SkipValidation,
    [switch]$ForceRecreate,
    [string]$EnvName = "ceramic-armor-ml",
    [string]$PythonVersion = "3.11"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to check system requirements
function Test-SystemRequirements {
    Write-ColorOutput "Checking system requirements..." "Yellow"
    
    # Check if conda is available
    try {
        $condaVersion = conda --version 2>$null
        Write-ColorOutput "  ✓ Conda found: $condaVersion" "Green"
    }
    catch {
        Write-ColorOutput "  ✗ Conda not found. Please install Miniconda or Anaconda first." "Red"
        Write-ColorOutput "    Download from: https://docs.conda.io/en/latest/miniconda.html" "Yellow"
        exit 1
    }
    
    # Check available disk space (require at least 5GB)
    $drive = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }
    $freeSpaceGB = [math]::Round($drive.FreeSpace / 1GB, 2)
    
    if ($freeSpaceGB -lt 5) {
        Write-ColorOutput "  ⚠ Warning: Less than 5GB free space available ($freeSpaceGB GB)" "Yellow"
    } else {
        Write-ColorOutput "  ✓ Disk space: $freeSpaceGB GB available" "Green"
    }
    
    # Check memory
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $memoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
    
    if ($memoryGB -lt 16) {
        Write-ColorOutput "  ⚠ Warning: Less than 16GB RAM detected ($memoryGB GB)" "Yellow"
    } else {
        Write-ColorOutput "  ✓ Memory: $memoryGB GB RAM" "Green"
    }
    
    # Check CPU cores
    $cpu = Get-WmiObject -Class Win32_Processor
    $cores = $cpu.NumberOfLogicalProcessors
    Write-ColorOutput "  ✓ CPU: $cores logical processors" "Green"
    
    Write-ColorOutput "System requirements check completed." "Green"
}

# Function to create conda environment
function New-CondaEnvironment {
    param(
        [string]$Name,
        [string]$PythonVer
    )
    
    Write-ColorOutput "Setting up conda environment '$Name'..." "Yellow"
    
    # Check if environment already exists
    $existingEnvs = conda env list | Select-String $Name
    
    if ($existingEnvs -and !$ForceRecreate) {
        Write-ColorOutput "  Environment '$Name' already exists." "Yellow"
        $choice = Read-Host "Do you want to recreate it? (y/N)"
        if ($choice -eq 'y' -or $choice -eq 'Y') {
            Write-ColorOutput "  Removing existing environment..." "Yellow"
            conda env remove -n $Name -y
        } else {
            Write-ColorOutput "  Using existing environment." "Green"
            return
        }
    } elseif ($existingEnvs -and $ForceRecreate) {
        Write-ColorOutput "  Force recreating environment..." "Yellow"
        conda env remove -n $Name -y
    }
    
    # Create new environment
    Write-ColorOutput "  Creating conda environment with Python $PythonVer..." "Yellow"
    conda create -n $Name python=$PythonVer -y
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "  ✗ Failed to create conda environment" "Red"
        exit 1
    }
    
    Write-ColorOutput "  ✓ Conda environment created successfully" "Green"
}

# Function to install Python packages
function Install-PythonPackages {
    Write-ColorOutput "Installing Python dependencies..." "Yellow"
    
    # Activate environment
    conda activate $EnvName
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "  ✗ Failed to activate conda environment" "Red"
        exit 1
    }
    
    # Upgrade pip
    Write-ColorOutput "  Upgrading pip..." "Yellow"
    python -m pip install --upgrade pip setuptools wheel
    
    # Install requirements
    Write-ColorOutput "  Installing packages from requirements.txt..." "Yellow"
    Write-ColorOutput "  This may take 10-15 minutes..." "Yellow"
    
    pip install -r requirements.txt --verbose
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "  ✗ Failed to install Python dependencies" "Red"
        Write-ColorOutput "    Common solutions:" "Yellow"
        Write-ColorOutput "    1. Check internet connection" "Yellow"
        Write-ColorOutput "    2. Install Visual Studio Build Tools" "Yellow"
        Write-ColorOutput "    3. Run PowerShell as Administrator" "Yellow"
        exit 1
    }
    
    Write-ColorOutput "  ✓ Python dependencies installed successfully" "Green"
}

# Function to create directory structure
function New-DirectoryStructure {
    Write-ColorOutput "Creating directory structure..." "Yellow"
    
    $directories = @(
        "data\raw",
        "data\processed", 
        "data\features",
        "data\splits",
        "data\data_collection",
        "results\models",
        "results\predictions",
        "results\metrics",
        "results\figures", 
        "results\reports",
        "logs",
        "notebooks"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "  ✓ Created: $dir" "Green"
        } else {
            Write-ColorOutput "  ✓ Exists: $dir" "Gray"
        }
    }
    
    Write-ColorOutput "Directory structure created successfully." "Green"
}

# Function to generate configuration files
function New-ConfigurationFiles {
    Write-ColorOutput "Generating configuration files..." "Yellow"
    
    # Create api_keys.yaml if it doesn't exist
    if (!(Test-Path "config\api_keys.yaml")) {
        Write-ColorOutput "  Creating config\api_keys.yaml from template..." "Yellow"
        Copy-Item "config\api_keys.yaml.example" "config\api_keys.yaml"
        Write-ColorOutput "  ⚠ IMPORTANT: Edit config\api_keys.yaml with your actual API keys" "Yellow"
    } else {
        Write-ColorOutput "  ✓ config\api_keys.yaml already exists" "Green"
    }
    
    # Create .env file
    if (!(Test-Path ".env")) {
        Write-ColorOutput "  Creating .env file..." "Yellow"
        
        $envContent = @"
# Environment variables for Ceramic Armor ML Project
# Intel CPU Optimization
OMP_NUM_THREADS=20
MKL_NUM_THREADS=20
NUMEXPR_NUM_THREADS=20
OPENBLAS_NUM_THREADS=20

# Project paths
PROJECT_ROOT=$PWD
DATA_PATH=$PWD\data
RESULTS_PATH=$PWD\results
CONFIG_PATH=$PWD\config

# API Configuration
MP_API_KEY=YOUR_MATERIALS_PROJECT_API_KEY_HERE
"@
        
        $envContent | Out-File -FilePath ".env" -Encoding UTF8
        Write-ColorOutput "  ✓ Created .env file" "Green"
    } else {
        Write-ColorOutput "  ✓ .env file already exists" "Green"
    }
    
    # Create PowerShell utility scripts
    Write-ColorOutput "  Creating PowerShell utility scripts..." "Yellow"
    
    # Activate environment script
    $activateScript = @"
# Activate the ceramic-armor-ml conda environment
try {
    conda activate ceramic-armor-ml
    Write-Host "Environment activated successfully" -ForegroundColor Green
    Write-Host "You can now run Python scripts in this environment" -ForegroundColor Yellow
} catch {
    Write-Host "ERROR: Failed to activate environment" -ForegroundColor Red
    Write-Host "Try running: conda activate ceramic-armor-ml" -ForegroundColor Yellow
}
"@
    
    $activateScript | Out-File -FilePath "activate_env.ps1" -Encoding UTF8
    
    # Quick test script
    $testScript = @"
# Quick test of the ceramic armor ML setup
try {
    conda activate ceramic-armor-ml
    Write-Host "Running setup validation..." -ForegroundColor Yellow
    python scripts\00_validate_setup.py
} catch {
    Write-Host "ERROR: Failed to run validation" -ForegroundColor Red
}
Read-Host "Press Enter to continue..."
"@
    
    $testScript | Out-File -FilePath "quick_test.ps1" -Encoding UTF8
    
    # Minimal test script
    $minimalScript = @"
# Run minimal test pipeline
try {
    conda activate ceramic-armor-ml
    Write-Host "Running minimal test pipeline..." -ForegroundColor Yellow
    python scripts\run_minimal_test.py
} catch {
    Write-Host "ERROR: Failed to run minimal test" -ForegroundColor Red
}
Read-Host "Press Enter to continue..."
"@
    
    $minimalScript | Out-File -FilePath "run_minimal_test.ps1" -Encoding UTF8
    
    Write-ColorOutput "Configuration files created successfully." "Green"
}

# Function to validate setup
function Test-Setup {
    if ($SkipValidation) {
        Write-ColorOutput "Skipping validation (as requested)." "Yellow"
        return
    }
    
    Write-ColorOutput "Validating setup..." "Yellow"
    
    # Test core imports
    Write-ColorOutput "  Testing Python imports..." "Yellow"
    
    $testImports = @"
try:
    import numpy, pandas, sklearn, xgboost, catboost, shap, pymatgen
    print('✓ Core imports: OK')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"@
    
    $result = python -c $testImports 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "  ✓ Core imports: OK" "Green"
    } else {
        Write-ColorOutput "  ✗ Core imports failed: $result" "Red"
    }
    
    # Test Intel optimizations
    $testIntel = @"
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print('✓ Intel optimizations: OK')
except ImportError:
    print('⚠ Intel optimizations not available')
"@
    
    $result = python -c $testIntel 2>&1
    Write-ColorOutput "  $result" "Yellow"
    
    # Test project imports
    $testProject = @"
import sys
sys.path.append('src')
try:
    import utils, data_collection
    print('✓ Project imports: OK')
except ImportError as e:
    print(f'⚠ Project imports: {e}')
"@
    
    $result = python -c $testProject 2>&1
    Write-ColorOutput "  $result" "Yellow"
    
    Write-ColorOutput "Setup validation completed." "Green"
}

# Function to display completion message
function Show-CompletionMessage {
    Write-ColorOutput "" "White"
    Write-ColorOutput "============================================================================" "Green"
    Write-ColorOutput "SETUP COMPLETE!" "Green"
    Write-ColorOutput "============================================================================" "Green"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Next steps:" "Yellow"
    Write-ColorOutput "  1. Edit config\api_keys.yaml with your Materials Project API key" "White"
    Write-ColorOutput "  2. Get your API key at: https://materialsproject.org/api" "White"
    Write-ColorOutput "  3. Run .\quick_test.ps1 to validate your setup" "White"
    Write-ColorOutput "  4. Run .\run_minimal_test.ps1 to test the pipeline" "White"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Useful PowerShell scripts:" "Yellow"
    Write-ColorOutput "  - .\activate_env.ps1        : Activate the conda environment" "White"
    Write-ColorOutput "  - .\quick_test.ps1          : Run setup validation" "White"
    Write-ColorOutput "  - .\run_minimal_test.ps1    : Run minimal test pipeline" "White"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Environment name: $EnvName" "White"
    Write-ColorOutput "To manually activate: conda activate $EnvName" "White"
    Write-ColorOutput "" "White"
    Write-ColorOutput "For troubleshooting, see docs\windows_troubleshooting.md" "White"
    Write-ColorOutput "" "White"
    
    # Create setup completion marker
    $completionInfo = @"
Setup completed on $(Get-Date)
Environment: $EnvName
Python version: $(python --version 2>&1)
PowerShell version: $($PSVersionTable.PSVersion)
"@
    
    $completionInfo | Out-File -FilePath "setup_complete.txt" -Encoding UTF8
}

# Main execution
try {
    Write-ColorOutput "" "White"
    Write-ColorOutput "============================================================================" "Cyan"
    Write-ColorOutput "CERAMIC ARMOR ML PROJECT - WINDOWS POWERSHELL SETUP" "Cyan"
    Write-ColorOutput "============================================================================" "Cyan"
    Write-ColorOutput "" "White"
    
    # Check if running as administrator
    if (Test-Administrator) {
        Write-ColorOutput "Running as Administrator: OK" "Green"
    } else {
        Write-ColorOutput "Not running as Administrator (some features may be limited)" "Yellow"
    }
    
    # Execute setup steps
    Test-SystemRequirements
    New-CondaEnvironment -Name $EnvName -PythonVer $PythonVersion
    Install-PythonPackages
    New-DirectoryStructure
    New-ConfigurationFiles
    Test-Setup
    Show-CompletionMessage
    
} catch {
    Write-ColorOutput "" "White"
    Write-ColorOutput "============================================================================" "Red"
    Write-ColorOutput "SETUP FAILED!" "Red"
    Write-ColorOutput "============================================================================" "Red"
    Write-ColorOutput "Error: $($_.Exception.Message)" "Red"
    Write-ColorOutput "" "White"
    Write-ColorOutput "For troubleshooting help:" "Yellow"
    Write-ColorOutput "  1. Check docs\windows_troubleshooting.md" "White"
    Write-ColorOutput "  2. Run with -Verbose for more details" "White"
    Write-ColorOutput "  3. Try running as Administrator" "White"
    Write-ColorOutput "" "White"
    exit 1
}

# Keep PowerShell window open
if ($Host.Name -eq "ConsoleHost") {
    Write-ColorOutput "Press any key to exit..." "Gray"
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}