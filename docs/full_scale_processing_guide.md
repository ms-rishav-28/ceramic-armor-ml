# Full-Scale Dataset Processing Guide

## Overview

This guide provides comprehensive instructions for processing the complete dataset of 5,600+ ceramic materials with full reproducibility and zero tolerance for approximation. The pipeline implements exact modeling strategies, mandatory feature engineering, and strict performance targets suitable for top-tier journal publication.

## System Requirements

### Hardware Requirements
- **CPU:** Multi-core processor (Intel recommended for optimizations)
- **Cores:** 8+ cores recommended (20 threads optimal)
- **RAM:** 16GB minimum, 32GB recommended for full dataset
- **Storage:** 10GB free space for data and intermediate files
- **Network:** Stable internet connection for API access

### Software Requirements
- **Python:** 3.8 or higher
- **Operating System:** Windows 10/11, Linux, or macOS
- **Package Manager:** pip or conda

### Required Python Packages
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
catboost>=1.1.0
requests>=2.28.0
tqdm>=4.64.0
pyyaml>=6.0
psutil>=5.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ceramic-armor-ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration Setup

Create the API keys configuration file:

```bash
# Create config directory if it doesn't exist
mkdir -p config

# Create API keys file
cat > config/api_keys.yaml << EOF
materials_project: "your_materials_project_api_key_here"
# Add other API keys as needed
EOF
```

### 3. Verify Installation

```bash
python scripts/verify_setup.py
```

## Quick Start

### Complete Pipeline Execution

```bash
# Execute the complete pipeline with default settings
python scripts/run_full_scale_processing.py

# Execute with custom parameters
python scripts/run_full_scale_processing.py \
    --output-dir data/processed/full_scale \
    --max-workers 8 \
    --batch-size 100 \
    --force-recollect
```

### Python API Usage

```python
from src.pipeline.full_scale_processor import FullScaleProcessor

# Initialize processor
processor = FullScaleProcessor(
    output_dir="data/processed/full_scale",
    max_workers=4,
    batch_size=100
)

# Process complete dataset
results = processor.process_full_dataset(
    force_recollect=False,
    validate_results=True,
    generate_reports=True
)

# Check results
if results['status'] == 'success':
    print(f"✓ Successfully processed {results['statistics']['total_materials_collected']} materials")
else:
    print(f"✗ Processing failed: {results['error']}")
```

## Processing Pipeline Overview

### Phase 1: Data Collection
- **Materials Project:** Primary source (3,500+ materials expected)
- **JARVIS-DFT:** Secondary source (1,000+ materials expected)
- **AFLOW:** Tertiary source (800+ materials expected)
- **NIST:** Specialized properties (300+ materials expected)

### Phase 2: Data Cleaning and Preprocessing
- Remove duplicates based on formula and structure
- Handle missing values with domain-specific strategies
- Standardize units (GPa for mechanical, W/m·K for thermal)
- Validate property ranges and flag outliers

### Phase 3: Feature Engineering
- **Mandatory Derived Properties:**
  - Specific Hardness = Hardness / Density
  - Brittleness Index = Hardness / Fracture Toughness
  - Ballistic Efficiency = Compressive Strength × √Hardness
  - Thermal Shock Resistance (complex thermal index)
  - Pugh Ratio = Shear Modulus / Bulk Modulus
  - Cauchy Pressure = C12 - C44

- **Compositional Features:** 40+ features
- **Structural Features:** 30+ features
- **Phase Stability Features:** 20+ features
- **Target:** 120+ total engineered features

### Phase 4: Data Validation
- Verify 5,600+ materials collected
- Ensure 120+ features generated
- Validate <5% missing values overall
- Check all ceramic systems represented
- Confirm all derived properties calculated

### Phase 5: Results Export
- CSV format (primary)
- Parquet format (efficient)
- Pickle format (preserves types)
- JSON metadata
- Processing statistics

## Configuration

### Key Configuration Files

1. **`config/config.yaml`** - Main pipeline configuration
2. **`config/model_params.yaml`** - Model hyperparameters
3. **`config/api_keys.yaml`** - API credentials

### Critical Settings (DO NOT MODIFY)

```yaml
# Performance targets (NON-NEGOTIABLE)
targets:
  mechanical_r2: 0.85
  ballistic_r2: 0.80

# Ceramic systems (REQUIRED)
ceramic_systems:
  primary: [SiC, Al2O3, B4C, WC, TiC]

# Derived properties (MANDATORY)
features:
  derived:
    - specific_hardness
    - brittleness_index
    - ballistic_efficacy
    - thermal_shock_resistance
    - pugh_ratio
    - cauchy_pressure
```

## Expected Outputs

### File Structure
```
data/processed/full_scale/
├── final_ceramic_materials_dataset.csv     # Main dataset (5,600+ materials)
├── final_ceramic_materials_dataset.parquet # Efficient format
├── final_ceramic_materials_dataset.pkl     # Python format
├── dataset_metadata.json                   # Dataset information
├── processing_statistics.json              # Processing stats
├── features/
│   ├── comprehensive_features.csv          # Feature matrix
│   └── feature_descriptions.json           # Feature documentation
├── processed/
│   └── cleaned_materials_data.csv          # Cleaned intermediate data
├── raw/
│   └── combined_materials_data.csv         # Raw combined data
├── reports/
│   ├── data_summary_report.md              # Data summary
│   ├── feature_analysis_report.md          # Feature analysis
│   ├── reproducibility_guide.md            # Reproducibility guide
│   └── execution_instructions.md           # Execution instructions
└── intermediate/                           # Source-specific data
    ├── SiC_materials.json
    ├── Al2O3_materials.json
    └── ...
```

### Dataset Specifications

- **Total Materials:** 5,600+ (target achieved)
- **Ceramic Systems:** 5 (SiC, Al₂O₃, B₄C, WC, TiC)
- **Total Features:** 120+ engineered features
- **Derived Properties:** 6 mandatory properties calculated
- **Data Quality:** <5% missing values overall
- **Format:** CSV, Parquet, and Pickle formats

## Validation and Quality Assurance

### Automated Validation Checks

```python
import pandas as pd

# Load and validate dataset
df = pd.read_csv('data/processed/full_scale/final_ceramic_materials_dataset.csv')

# Validate material count
assert len(df) >= 5600, f"Expected 5600+ materials, got {len(df)}"

# Validate feature count
assert df.shape[1] >= 120, f"Expected 120+ features, got {df.shape[1]}"

# Validate ceramic systems
expected_systems = {'SiC', 'Al2O3', 'B4C', 'WC', 'TiC'}
actual_systems = set(df['ceramic_system'].unique())
assert actual_systems == expected_systems, f"System mismatch: {actual_systems}"

# Validate derived properties
required_props = [
    'specific_hardness', 'brittleness_index', 'ballistic_efficiency',
    'thermal_shock_resistance', 'pugh_ratio', 'cauchy_pressure'
]
for prop in required_props:
    assert prop in df.columns, f"Missing required property: {prop}"
    assert df[prop].notna().sum() > 0, f"No valid values for {prop}"

print("✓ All validation checks passed")
```

### Data Quality Metrics

- **Completeness:** >95% for primary features
- **Consistency:** Standardized units and naming
- **Accuracy:** Validated against known material properties
- **Coverage:** All 5 ceramic systems represented
- **Uniqueness:** Duplicates removed based on formula and structure

## Troubleshooting

### Common Issues and Solutions

#### 1. API Rate Limiting
**Problem:** Materials Project API rate limits exceeded
**Solution:** 
- Increase delays in collector configuration
- Use existing cached data with `force_recollect=False`
- Check API key validity

#### 2. Memory Issues
**Problem:** Out of memory during processing
**Solution:**
- Reduce `batch_size` parameter (try 50 or 25)
- Reduce `max_workers` (try 2 or 1)
- Close other applications
- Use system with more RAM

#### 3. Network Connectivity
**Problem:** Network timeouts or connection errors
**Solution:**
- Check internet connection stability
- Increase timeout values in configuration
- Use cached intermediate results
- Retry processing with existing data

#### 4. Missing Dependencies
**Problem:** Import errors or missing packages
**Solution:**
```bash
pip install -r requirements.txt --upgrade
pip install --force-reinstall pandas numpy scikit-learn
```

#### 5. Configuration Errors
**Problem:** Invalid configuration values
**Solution:**
- Verify `config/config.yaml` syntax
- Check API keys in `config/api_keys.yaml`
- Use default configuration if custom config fails

### Error Recovery

The pipeline includes automatic error recovery:
- **Intermediate Saves:** Results saved at each major step
- **Resume Capability:** Can resume from last successful checkpoint
- **Error Logging:** Detailed error logs in `logs/` directory
- **Graceful Degradation:** Continues processing even if some materials fail

### Validation Commands

```bash
# Validate existing data without reprocessing
python scripts/run_full_scale_processing.py --validate-only

# Check system requirements
python scripts/verify_setup.py

# Test API connectivity
python scripts/test_api_connectivity.py

# Validate configuration
python -c "from src.utils.config_loader import load_project_config; print('✓ Config valid')"
```

## Performance Optimization

### CPU Optimization

```python
# Enable Intel optimizations (if available)
import os
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'

# Use Intel Extension for Scikit-learn
from src.utils.intel_optimizer import IntelOptimizer
optimizer = IntelOptimizer()
optimizer.configure_environment()
```

### Memory Optimization

```python
# Use memory-efficient processing
processor = FullScaleProcessor(
    batch_size=50,      # Smaller batches
    max_workers=2,      # Fewer workers
    enable_parallel=False  # Disable parallelism if needed
)
```

### Storage Optimization

- Use Parquet format for large datasets (smaller file size)
- Enable compression for intermediate files
- Clean up intermediate files after successful processing

## Reproducibility Guidelines

### Deterministic Processing

```python
# Set random seeds for reproducibility
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
```

### Version Control

- Pin all package versions in `requirements.txt`
- Document Python version used
- Save configuration files with results
- Record processing timestamps

### Independent Verification

```python
# Verification script for independent execution
def verify_reproducibility():
    """Verify that results can be reproduced independently."""
    
    # Load dataset
    df = pd.read_csv('data/processed/full_scale/final_ceramic_materials_dataset.csv')
    
    # Load metadata
    with open('data/processed/full_scale/dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Verify counts match metadata
    assert len(df) == metadata['total_materials']
    assert df.shape[1] == metadata['total_features']
    
    # Verify derived properties
    derived_props = ['specific_hardness', 'brittleness_index', 'ballistic_efficiency']
    for prop in derived_props:
        assert prop in df.columns
        # Verify calculation (example for specific_hardness)
        if prop == 'specific_hardness' and 'hardness' in df.columns and 'density' in df.columns:
            calculated = df['hardness'] / df['density']
            np.testing.assert_allclose(df[prop].dropna(), calculated.dropna(), rtol=1e-10)
    
    print("✓ Reproducibility verification passed")

# Run verification
verify_reproducibility()
```

## Next Steps

After successful processing:

1. **Load the Dataset:**
   ```python
   import pandas as pd
   df = pd.read_csv('data/processed/full_scale/final_ceramic_materials_dataset.csv')
   ```

2. **Review Feature Documentation:**
   ```python
   import json
   with open('data/processed/full_scale/features/feature_descriptions.json', 'r') as f:
       feature_docs = json.load(f)
   ```

3. **Train Models:** Use exact modeling strategy (XGBoost, CatBoost, Random Forest, Gradient Boosting)

4. **Validate Performance:** Ensure R² ≥ 0.85 (mechanical) and R² ≥ 0.80 (ballistic)

5. **Generate Interpretability:** Use SHAP analysis for mechanistic insights

## Support and Contact

For issues or questions:
1. Check the troubleshooting section above
2. Review error logs in `logs/` directory
3. Validate system requirements and configuration
4. Ensure all dependencies are correctly installed

## Version Information

- **Pipeline Version:** 1.0.0
- **Minimum Python:** 3.8
- **Target Materials:** 5,600+
- **Target Features:** 120+
- **Performance Targets:** R² ≥ 0.85 (mechanical), R² ≥ 0.80 (ballistic)

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Documentation Version:** 1.0.0