# Windows Troubleshooting Guide - Ceramic Armor ML Project

## Overview

This guide provides detailed troubleshooting solutions for common issues encountered when running the Ceramic Armor ML project on Windows systems.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Environment Issues](#environment-issues)
3. [Import and Module Issues](#import-and-module-issues)
4. [API and Network Issues](#api-and-network-issues)
5. [Performance Issues](#performance-issues)
6. [File System Issues](#file-system-issues)
7. [Memory and Resource Issues](#memory-and-resource-issues)
8. [Advanced Troubleshooting](#advanced-troubleshooting)

## Installation Issues

### Problem: Conda Installation Fails

**Symptoms**:
- `conda: command not found`
- Installation hangs or fails

**Diagnosis**:
```cmd
# Check if conda is in PATH
where conda

# Check environment variables
echo %PATH%
```

**Solutions**:

1. **Reinstall with PATH option**:
   ```cmd
   # Uninstall current conda
   # Download fresh installer
   # During installation, check "Add to PATH"
   ```

2. **Manual PATH addition**:
   ```cmd
   # Add to system PATH (requires admin)
   setx PATH "%PATH%;C:\Users\%USERNAME%\miniconda3\Scripts" /M
   ```

3. **Use conda prompt**:
   - Search for "Anaconda Prompt" in Start Menu
   - Use this instead of regular Command Prompt

### Problem: Python Package Installation Fails

**Symptoms**:
- `Microsoft Visual C++ 14.0 is required`
- `error: Microsoft Visual C++ 14.0 or greater is required`

**Diagnosis**:
```cmd
# Check if Visual Studio Build Tools are installed
where cl.exe
```

**Solutions**:

1. **Install Visual Studio Build Tools**:
   ```powershell
   # Download and install
   Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile "vs_buildtools.exe"
   .\vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows10SDK.19041
   ```

2. **Alternative: Install Visual Studio Community**:
   - Download from https://visualstudio.microsoft.com/
   - Install with C++ development tools

3. **Use pre-compiled wheels**:
   ```cmd
   # Force wheel installation
   pip install --only-binary=all -r requirements.txt
   ```

### Problem: Long Installation Times

**Symptoms**:
- Package installation takes hours
- Installation appears to hang

**Diagnosis**:
```cmd
# Check network connectivity
ping pypi.org

# Check available disk space
dir /-c
```

**Solutions**:

1. **Use conda-forge channel**:
   ```cmd
   conda install -c conda-forge numpy pandas scikit-learn
   ```

2. **Install in batches**:
   ```cmd
   # Install core packages first
   pip install numpy pandas scipy
   # Then ML packages
   pip install scikit-learn xgboost catboost
   # Finally specialized packages
   pip install pymatgen mp-api
   ```

3. **Use local package cache**:
   ```cmd
   # Download packages first
   pip download -r requirements.txt -d packages
   # Install from local cache
   pip install --find-links packages -r requirements.txt
   ```

## Environment Issues

### Problem: Environment Activation Fails

**Symptoms**:
- `conda activate` doesn't work
- Environment not found

**Diagnosis**:
```cmd
# List all environments
conda env list

# Check current environment
conda info --envs
```

**Solutions**:

1. **Initialize conda for cmd**:
   ```cmd
   conda init cmd.exe
   # Restart command prompt
   ```

2. **Use full path**:
   ```cmd
   C:\Users\%USERNAME%\miniconda3\Scripts\activate ceramic-armor-ml
   ```

3. **Recreate environment**:
   ```cmd
   conda env remove -n ceramic-armor-ml
   conda create -n ceramic-armor-ml python=3.11 -y
   ```

### Problem: Wrong Python Version in Environment

**Symptoms**:
- Python version doesn't match expected
- Package compatibility issues

**Diagnosis**:
```cmd
conda activate ceramic-armor-ml
python --version
which python
```

**Solutions**:

1. **Specify Python version explicitly**:
   ```cmd
   conda install python=3.11 -n ceramic-armor-ml
   ```

2. **Recreate with specific version**:
   ```cmd
   conda create -n ceramic-armor-ml python=3.11.5 -y
   ```

## Import and Module Issues

### Problem: ModuleNotFoundError

**Symptoms**:
- `ModuleNotFoundError: No module named 'src'`
- `ImportError: cannot import name 'X' from 'Y'`

**Diagnosis**:
```cmd
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check if module exists
python -c "import src; print(src.__file__)"
```

**Solutions**:

1. **Add project root to PYTHONPATH**:
   ```cmd
   set PYTHONPATH=%cd%;%PYTHONPATH%
   ```

2. **Install project in development mode**:
   ```cmd
   pip install -e .
   ```

3. **Fix __init__.py files**:
   ```cmd
   # Ensure all directories have __init__.py
   type nul > src\__init__.py
   type nul > src\utils\__init__.py
   ```

### Problem: Intel Optimization Import Fails

**Symptoms**:
- `ImportError: No module named 'sklearnex'`
- Intel optimizations not working

**Diagnosis**:
```cmd
# Check if Intel extension is installed
pip show scikit-learn-intelex
```

**Solutions**:

1. **Install Intel extension**:
   ```cmd
   pip install scikit-learn-intelex
   ```

2. **Check compatibility**:
   ```cmd
   # Verify Intel CPU
   wmic cpu get name
   ```

3. **Fallback without Intel optimization**:
   ```python
   # In code, add try/except
   try:
       from sklearnex import patch_sklearn
       patch_sklearn()
   except ImportError:
       print("Intel optimizations not available")
   ```

## API and Network Issues

### Problem: Materials Project API Fails

**Symptoms**:
- `API key not found`
- `Connection timeout`
- `Rate limit exceeded`

**Diagnosis**:
```cmd
# Test API connectivity
python scripts\test_api_connectivity.py

# Check API key format
type config\api_keys.yaml
```

**Solutions**:

1. **Verify API key**:
   ```yaml
   # In config/api_keys.yaml
   materials_project: "mp-1234567890abcdef"  # Correct format
   ```

2. **Test API manually**:
   ```python
   from mp_api.client import MPRester
   with MPRester("your-api-key") as mpr:
       data = mpr.summary.search(formula="SiC")
       print(f"Found {len(data)} entries")
   ```

3. **Handle rate limiting**:
   ```python
   # Add delays between requests
   import time
   time.sleep(1)  # 1 second delay
   ```

### Problem: Network Proxy Issues

**Symptoms**:
- `ProxyError`
- `SSL certificate verify failed`

**Diagnosis**:
```cmd
# Check proxy settings
netsh winhttp show proxy
```

**Solutions**:

1. **Configure pip for proxy**:
   ```cmd
   pip config set global.proxy http://proxy.company.com:8080
   ```

2. **Disable SSL verification (not recommended)**:
   ```cmd
   pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
   ```

3. **Use corporate certificates**:
   ```cmd
   # Add corporate CA bundle
   set REQUESTS_CA_BUNDLE=C:\path\to\corporate-ca-bundle.crt
   ```

## Performance Issues

### Problem: Slow Training Performance

**Symptoms**:
- Training takes much longer than expected
- High CPU usage but low utilization

**Diagnosis**:
```cmd
# Check CPU usage
wmic cpu get loadpercentage /value

# Check memory usage
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value
```

**Solutions**:

1. **Optimize thread settings**:
   ```cmd
   # Set environment variables
   set OMP_NUM_THREADS=20
   set MKL_NUM_THREADS=20
   set NUMEXPR_NUM_THREADS=20
   ```

2. **Enable Intel optimizations**:
   ```python
   from sklearnex import patch_sklearn
   patch_sklearn()
   ```

3. **Reduce dataset size for testing**:
   ```yaml
   # In config/config.yaml
   data_sources:
     materials_project:
       max_entries: 1000  # Reduce for testing
   ```

### Problem: Memory Usage Too High

**Symptoms**:
- `MemoryError`
- System becomes unresponsive
- Swap file usage high

**Diagnosis**:
```cmd
# Check memory usage
tasklist /fi "imagename eq python.exe" /fo table

# Check virtual memory settings
wmic pagefile list /format:list
```

**Solutions**:

1. **Increase virtual memory**:
   - Control Panel > System > Advanced > Performance Settings
   - Advanced > Virtual Memory > Change
   - Set custom size: Initial 16GB, Maximum 32GB

2. **Use memory mapping**:
   ```python
   import numpy as np
   # Use memory-mapped arrays for large datasets
   data = np.memmap('data.dat', dtype='float32', mode='r')
   ```

3. **Process data in chunks**:
   ```python
   # Process data in smaller batches
   chunk_size = 1000
   for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

## File System Issues

### Problem: Path Length Limitations

**Symptoms**:
- `FileNotFoundError` with long paths
- `OSError: [Errno 36] File name too long`

**Diagnosis**:
```cmd
# Check current path length
echo %cd% | find /c /v ""
```

**Solutions**:

1. **Enable long path support**:
   ```cmd
   # Run as administrator
   reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
   ```

2. **Use shorter paths**:
   ```cmd
   # Move project to shorter path
   move C:\very\long\path\to\project C:\ceramic-ml
   ```

3. **Use pathlib for path handling**:
   ```python
   from pathlib import Path
   # Handles long paths better than os.path
   path = Path("data") / "processed" / "features.csv"
   ```

### Problem: Permission Denied Errors

**Symptoms**:
- `PermissionError: [Errno 13] Permission denied`
- Cannot write to directories

**Diagnosis**:
```cmd
# Check file permissions
icacls data\
```

**Solutions**:

1. **Run as administrator**:
   - Right-click Command Prompt
   - Select "Run as administrator"

2. **Change directory permissions**:
   ```cmd
   # Give full control to current user
   icacls data\ /grant %USERNAME%:F /T
   ```

3. **Check antivirus interference**:
   - Add project folder to antivirus exclusions
   - Temporarily disable real-time protection

## Memory and Resource Issues

### Problem: Out of Memory During Training

**Symptoms**:
- `MemoryError`
- Python process killed
- System freezes

**Diagnosis**:
```powershell
# Check available memory
Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory

# Monitor memory usage during training
Get-Process python | Select-Object ProcessName,WorkingSet,PagedMemorySize
```

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   # In config/config.yaml
   training:
     batch_size: 500  # Reduce from default
   ```

2. **Use incremental learning**:
   ```python
   # For scikit-learn models
   from sklearn.linear_model import SGDRegressor
   model = SGDRegressor()
   for chunk in data_chunks:
       model.partial_fit(chunk)
   ```

3. **Enable memory monitoring**:
   ```python
   import psutil
   import os
   
   def check_memory():
       process = psutil.Process(os.getpid())
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Memory usage: {memory_mb:.1f} MB")
   ```

### Problem: Disk Space Issues

**Symptoms**:
- `OSError: [Errno 28] No space left on device`
- Cannot save results

**Diagnosis**:
```cmd
# Check disk space
dir /-c

# Check specific directories
dir data\ /-c
dir results\ /-c
```

**Solutions**:

1. **Clean temporary files**:
   ```cmd
   # Clean conda cache
   conda clean --all

   # Clean pip cache
   pip cache purge

   # Clean Windows temp files
   del /q /s %TEMP%\*
   ```

2. **Move data to different drive**:
   ```cmd
   # Move data directory
   move data D:\ceramic-ml-data
   mklink /D data D:\ceramic-ml-data
   ```

3. **Compress old results**:
   ```powershell
   # Compress old results
   Compress-Archive -Path "results\old_*" -DestinationPath "results\archive.zip"
   Remove-Item "results\old_*" -Recurse
   ```

## Advanced Troubleshooting

### Problem: Debugging Import Issues

**Symptoms**:
- Complex import errors
- Circular imports
- Module conflicts

**Diagnosis Tools**:

1. **Import tracer**:
   ```cmd
   python -v -c "import src.utils"
   ```

2. **Module finder**:
   ```python
   import importlib.util
   spec = importlib.util.find_spec("src.utils")
   print(spec.origin if spec else "Module not found")
   ```

3. **Dependency analysis**:
   ```cmd
   pip show --verbose scikit-learn
   ```

### Problem: Performance Profiling

**Tools for performance analysis**:

1. **Memory profiler**:
   ```cmd
   pip install memory-profiler
   python -m memory_profiler scripts\run_minimal_test.py
   ```

2. **CPU profiler**:
   ```cmd
   pip install py-spy
   py-spy record -o profile.svg -- python scripts\run_minimal_test.py
   ```

3. **Line profiler**:
   ```cmd
   pip install line-profiler
   kernprof -l -v scripts\run_minimal_test.py
   ```

### Problem: Environment Debugging

**Comprehensive environment check**:

```cmd
# Create debug script
echo @echo off > debug_env.bat
echo echo ===== ENVIRONMENT DEBUG ===== >> debug_env.bat
echo echo Python version: >> debug_env.bat
echo python --version >> debug_env.bat
echo echo Conda info: >> debug_env.bat
echo conda info >> debug_env.bat
echo echo Installed packages: >> debug_env.bat
echo pip list >> debug_env.bat
echo echo Environment variables: >> debug_env.bat
echo set ^| findstr /i "python\|conda\|path" >> debug_env.bat
echo echo System info: >> debug_env.bat
echo systeminfo ^| findstr /i "memory\|processor" >> debug_env.bat

# Run debug script
debug_env.bat > debug_report.txt
```

## Emergency Recovery

### Complete Environment Reset

If everything fails, use this nuclear option:

```cmd
# 1. Remove conda environment
conda env remove -n ceramic-armor-ml

# 2. Clear conda cache
conda clean --all

# 3. Clear pip cache
pip cache purge

# 4. Remove project virtual environments
rmdir /s /q .venv

# 5. Reinstall from scratch
setup_windows.bat
```

### Backup and Restore

**Create backup before major changes**:

```cmd
# Backup environment
conda env export -n ceramic-armor-ml > environment_backup.yml

# Backup configuration
xcopy config config_backup\ /E /I

# Restore environment
conda env create -f environment_backup.yml
```

## Getting Additional Help

### Log Collection

Before seeking help, collect these logs:

```cmd
# System information
systeminfo > system_info.txt

# Environment information
conda info --envs > conda_info.txt
pip list > pip_list.txt

# Error logs
copy logs\*.log error_logs\

# Configuration files
copy config\*.yaml config_backup\
```

### Useful Commands for Support

```cmd
# Quick system check
python --version && conda --version && pip --version

# Package versions
pip show numpy pandas scikit-learn xgboost catboost

# Environment variables
set | findstr /i "python\|conda\|path\|omp\|mkl"

# Memory and CPU info
wmic computersystem get TotalPhysicalMemory
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors
```

This troubleshooting guide should help resolve most Windows-specific issues. For persistent problems, collect the diagnostic information above and consult the project documentation or support channels.