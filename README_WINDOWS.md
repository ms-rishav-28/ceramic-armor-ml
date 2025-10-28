# Ceramic Armor ML - Windows Setup

## Quick Start

This project provides machine learning models for predicting mechanical and ballistic properties of ceramic armor materials. This guide is specifically for Windows 11 Pro 64-bit systems.

### Prerequisites

1. **Install Miniconda** (recommended) or Anaconda:
   - Download: https://docs.conda.io/en/latest/miniconda.html
   - During installation, check "Add to PATH"

2. **Install Visual Studio Build Tools** (if needed):
   - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "C++ build tools" workload

### Automated Setup (Recommended)

Choose one of these setup methods:

#### Option 1: Batch Script (Simple)
```cmd
setup_windows.bat
```

#### Option 2: PowerShell Script (Advanced)
```powershell
# Run in PowerShell (may require execution policy change)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows.ps1
```

### Manual Setup (If Automated Fails)

1. **Create conda environment**:
   ```cmd
   conda create -n ceramic-armor-ml python=3.11 -y
   conda activate ceramic-armor-ml
   ```

2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   ```cmd
   copy config\api_keys.yaml.example config\api_keys.yaml
   notepad config\api_keys.yaml
   ```

4. **Get Materials Project API key**:
   - Visit: https://materialsproject.org/api
   - Create account and generate API key
   - Add to `config\api_keys.yaml`

### Validation

Test your setup:

```cmd
# Quick validation
conda activate ceramic-armor-ml
python scripts\00_validate_setup.py

# Minimal test pipeline (30 minutes)
python scripts\run_minimal_test.py
```

## Project Structure

```
ceramic-armor-ml/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   ├── model_params.yaml  # Model hyperparameters
│   └── api_keys.yaml      # API keys (create from .example)
├── data/                  # Data directories
│   ├── raw/              # Raw data from APIs
│   ├── processed/        # Cleaned data
│   └── features/         # Engineered features
├── src/                  # Source code
│   ├── data_collection/  # Data collectors
│   ├── models/          # ML models
│   ├── utils/           # Utilities
│   └── ...
├── scripts/             # Execution scripts
├── results/            # Model outputs
└── docs/              # Documentation
```

## Common Windows Issues

### Issue: "conda not found"
**Solution**: Reinstall Miniconda with PATH option checked, or use Anaconda Prompt

### Issue: "Microsoft Visual C++ required"
**Solution**: Install Visual Studio Build Tools with C++ workload

### Issue: Import errors
**Solution**: Ensure you're in the project root and environment is activated:
```cmd
cd C:\path\to\ceramic-armor-ml
conda activate ceramic-armor-ml
```

### Issue: API connection fails
**Solution**: Check your API key in `config\api_keys.yaml` and internet connection

### Issue: Memory errors
**Solution**: Increase virtual memory in Windows settings or reduce dataset size

## Performance Optimization

For Intel i7-12700K systems, the project includes optimizations:

- Intel scikit-learn extensions
- Multi-threading (20 threads)
- Optimized memory usage

Environment variables are set automatically:
```
OMP_NUM_THREADS=20
MKL_NUM_THREADS=20
NUMEXPR_NUM_THREADS=20
```

## Documentation

- **Complete Setup Guide**: `docs\windows_setup_guide.md`
- **Troubleshooting**: `docs\windows_troubleshooting.md`
- **API Reference**: `docs\api_reference.rst`
- **Quick Start**: `docs\quickstart.rst`

## Utility Scripts

After setup, use these Windows-specific scripts:

- `activate_env.bat` - Activate conda environment
- `quick_test.bat` - Run setup validation
- `run_minimal_test.bat` - Test pipeline (30 min)
- `activate_env.ps1` - PowerShell version
- `quick_test.ps1` - PowerShell validation
- `run_minimal_test.ps1` - PowerShell test

## System Requirements

### Minimum
- Windows 10/11 64-bit
- 16GB RAM
- 20GB free disk space
- Internet connection

### Recommended
- Windows 11 Pro 64-bit
- Intel i7-12700K (20 threads)
- 32GB RAM
- 50GB free SSD space
- Stable internet connection

## Getting Help

1. **Check logs**: `logs\ceramic_armor_ml.log`
2. **Run diagnostics**: `python scripts\verify_setup.py`
3. **Review troubleshooting**: `docs\windows_troubleshooting.md`

## Next Steps

After successful setup:

1. Configure your Materials Project API key
2. Run validation: `quick_test.bat`
3. Test pipeline: `run_minimal_test.bat`
4. Review documentation in `docs\`
5. Start with small datasets before full-scale runs

## Support

For Windows-specific issues:
- Check `docs\windows_troubleshooting.md`
- Ensure all prerequisites are installed
- Try running as Administrator
- Check antivirus exclusions

The project is optimized for Windows and includes comprehensive error handling and recovery mechanisms.