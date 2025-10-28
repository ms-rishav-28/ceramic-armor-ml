# Minimal Test Pipeline

This directory contains scripts for running a fast end-to-end test of the ceramic armor ML pipeline.

## Overview

The minimal test pipeline validates the complete ML pipeline using:
- 100 samples per ceramic system (SiC, Al2O3, B4C)
- Young's modulus prediction only (fastest property)
- Simplified model parameters for speed
- Target completion time: under 30 minutes

## Scripts

### 1. `run_minimal_test.py` (Recommended)
**Main entry point** - Runs the complete test and validation process.

```bash
python scripts/run_minimal_test.py
```

This script:
- Executes the minimal test pipeline
- Validates all results
- Generates comprehensive reports
- Provides pass/fail determination

### 2. `minimal_test_pipeline.py`
Core test pipeline implementation that:
- Collects data using Materials Project API
- Preprocesses data using existing pipeline
- Engineers features using existing modules
- Trains models with simplified parameters
- Generates basic test report

```bash
python scripts/minimal_test_pipeline.py
```

### 3. `validate_minimal_test.py`
Validation script that:
- Validates each pipeline stage
- Checks data quality and model performance
- Generates comprehensive validation report
- Provides automated pass/fail determination

```bash
python scripts/validate_minimal_test.py
```

## Prerequisites

1. **API Keys**: Ensure `config/api_keys.yaml` contains your Materials Project API key:
   ```yaml
   materials_project: "your_api_key_here"
   ```

2. **Dependencies**: All required packages should be installed (see `requirements.txt`)

3. **Configuration**: Ensure `config/config.yaml` is properly configured

## Output Files

The test creates the following directory structure under `data/test_pipeline/`:

```
data/test_pipeline/
├── raw/                          # Raw collected data
│   ├── sic_test_raw.csv
│   ├── al2o3_test_raw.csv
│   └── b4c_test_raw.csv
├── processed/                    # Preprocessed data
│   ├── sic_test_processed.csv
│   ├── al2o3_test_processed.csv
│   └── b4c_test_processed.csv
├── features/                     # Feature-engineered data
│   ├── sic_test_features.csv
│   ├── al2o3_test_features.csv
│   └── b4c_test_features.csv
├── models/                       # Trained models
│   ├── sic/youngs_modulus/
│   ├── al2o3/youngs_modulus/
│   └── b4c/youngs_modulus/
├── minimal_test_report.txt       # Basic test report
└── validation_report.txt         # Comprehensive validation report
```

## Success Criteria

The test is considered successful if:
- ✅ At least 2 ceramic systems complete successfully
- ✅ Models achieve R² ≥ 0.5 (relaxed threshold for minimal test)
- ✅ Complete execution in under 30 minutes
- ✅ All pipeline stages complete without critical errors

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: Materials Project API key not found
   ```
   - Solution: Add your API key to `config/api_keys.yaml`

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src.xxx'
   ```
   - Solution: Run from project root directory
   - Ensure all dependencies are installed

3. **Timeout Issues**
   ```
   Error: Minimal test pipeline timed out
   ```
   - Solution: Check internet connectivity
   - Verify API key is valid and has sufficient quota

4. **Low Performance**
   ```
   Warning: Models achieve R² < 0.5
   ```
   - This may indicate data quality issues
   - Check the validation report for specific recommendations

### Getting Help

1. Check the generated reports in `data/test_pipeline/`
2. Review the log output for specific error messages
3. Run the setup validation script: `python scripts/00_validate_setup.py`
4. Verify API connectivity: `python scripts/test_api_connectivity.py`

## Performance Expectations

On a typical system with good internet connectivity:
- Data collection: 5-10 minutes
- Preprocessing: 1-2 minutes
- Feature engineering: 2-3 minutes
- Model training: 10-15 minutes
- Validation: 1-2 minutes
- **Total: 20-30 minutes**

## Next Steps

After a successful minimal test:
1. Review the validation report for any warnings
2. Run the full pipeline: `python scripts/run_full_pipeline.py`
3. Or continue with specific pipeline stages as needed