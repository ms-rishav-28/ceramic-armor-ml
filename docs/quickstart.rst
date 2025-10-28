Quick Start Guide
=================

This guide will get you up and running with the Ceramic Armor ML Pipeline in under 30 minutes.

Complete Pipeline Execution
----------------------------

Run the entire pipeline with one command:

.. code-block:: bash

   python scripts/run_full_pipeline.py

This executes all phases:

1. **Data Collection** (30-60 minutes)
   - Materials Project API calls
   - AFLOW AFLUX queries
   - JARVIS data download
   - NIST CSV loading
   - Literature mining (optional)

2. **Data Integration** (5-10 minutes)
   - Cross-source merging
   - Deduplication
   - Quality filtering

3. **Preprocessing** (10-15 minutes)
   - Unit standardization
   - Outlier detection and removal
   - Missing value imputation

4. **Feature Engineering** (15-30 minutes)
   - Compositional descriptors (30+ features)
   - Derived properties (Pugh ratio, specific properties)
   - Microstructure features (Hall-Petch terms)
   - Phase stability analysis

5. **Model Training** (4-8 hours)
   - XGBoost, CatBoost, Random Forest, Gradient Boosting
   - Hyperparameter optimization
   - Ensemble stacking
   - Cross-validation

6. **Evaluation & Interpretation** (30-60 minutes)
   - Performance metrics
   - SHAP analysis
   - Visualization generation

Step-by-Step Execution
----------------------

For more control, run individual phases:

**Phase 1: Data Collection**

.. code-block:: python

   from data.data_collection.aflow_collector import AFLOWCollector
   from data.data_collection.jarvis_collector import JARVISCollector
   
   # Collect AFLOW data
   aflow = AFLOWCollector()
   aflow.collect('SiC')
   
   # Collect JARVIS data
   jarvis = JARVISCollector()
   jarvis.collect('SiC')

**Phase 2: Preprocessing**

.. code-block:: python

   from src.preprocessing.unit_standardizer import standardize
   from src.preprocessing.outlier_detector import remove_iqr_outliers
   from src.preprocessing.missing_value_handler import impute_knn
   import pandas as pd
   
   df = pd.read_csv('data/processed/sic/sic_integrated.csv')
   df = standardize(df)
   df = remove_iqr_outliers(df, ['density', 'youngs_modulus'])
   df = impute_knn(df)

**Phase 3: Feature Engineering**

.. code-block:: python

   from src.feature_engineering.compositional_features import CompositionalFeatureCalculator
   from src.feature_engineering.microstructure_features import MicrostructureFeatureCalculator
   
   comp_calc = CompositionalFeatureCalculator()
   micro_calc = MicrostructureFeatureCalculator()
   
   df = comp_calc.augment_dataframe(df, formula_col='formula')
   df = micro_calc.add_features(df)

**Phase 4: Training**

.. code-block:: python

   from src.training.trainer import CeramicPropertyTrainer
   import yaml
   
   with open('config/config.yaml') as f:
       config = yaml.safe_load(f)
   
   trainer = CeramicPropertyTrainer(config)
   trainer.train_all_systems()

**Phase 5: Evaluation**

.. code-block:: bash

   python scripts/05_evaluate_models.py

**Phase 6: Interpretation**

.. code-block:: bash

   python scripts/06_interpret_results.py

Expected Results
----------------

After successful execution, you should see:

**Console Output:**
::

   ✓ Data collection complete: 15,000+ samples across 5 systems
   ✓ Preprocessing complete: Standardized units, removed outliers
   ✓ Feature engineering complete: 120+ features generated
   ✓ Training complete: All models trained with R² > 0.80
   ✓ SHAP analysis complete: Feature importance plots generated

**Generated Files:**
::

   results/
   ├── models/           # Trained model files (.pkl)
   ├── predictions/      # Test predictions (.csv)
   ├── figures/          # Parity plots, SHAP plots (.png)
   └── metrics/          # Performance metrics (.json)

**Performance Verification:**

.. code-block:: python

   import json
   
   # Check model performance
   with open('results/metrics/sic_vickers_hardness_metrics.json') as f:
       metrics = json.load(f)
   
   print(f"R² Score: {metrics['r2']:.3f}")
   print(f"RMSE: {metrics['rmse']:.2f}")
   # Expected: R² > 0.85 for mechanical properties

Troubleshooting
---------------

**Common Issues:**

1. **API Key Errors**
   - Verify ``config/api_keys.yaml`` exists and contains valid keys
   - Check Materials Project API quota

2. **Memory Issues**
   - Reduce batch sizes in ``config/config.yaml``
   - Close other applications during training

3. **Missing Data**
   - Ensure NIST CSV files are properly placed
   - Check internet connection for API calls

4. **Import Errors**
   - Verify all dependencies installed: ``pip install -r requirements.txt``
   - Check Python path includes project root

**Getting Help:**

- Check log files in ``logs/`` directory
- Enable debug logging: ``export LOGURU_LEVEL=DEBUG``
- Review error messages for specific module failures