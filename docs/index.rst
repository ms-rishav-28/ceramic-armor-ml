Ceramic Armor ML Pipeline Documentation
=====================================

Welcome to the Ceramic Armor ML Pipeline documentation. This system provides publication-grade machine learning for ballistic property prediction of ceramic armor materials.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   data_collection
   preprocessing
   feature_engineering
   training
   evaluation
   interpretation
   api_reference

Overview
--------

This pipeline implements a complete machine learning workflow for predicting mechanical and ballistic properties of ceramic armor materials including SiC, Al₂O₃, B₄C, WC, and TiC.

Key Features
------------

* **Multi-source data integration**: Materials Project, AFLOW, JARVIS, NIST
* **Advanced preprocessing**: Unit standardization, outlier detection, missing value imputation
* **Comprehensive feature engineering**: 120+ features including compositional, derived, and microstructure descriptors
* **Ensemble modeling**: XGBoost, CatBoost, Random Forest, Gradient Boosting with stacking
* **Cross-validation**: K-fold and Leave-One-Ceramic-Out validation
* **Hyperparameter optimization**: Optuna-based tuning
* **Model interpretation**: SHAP analysis and materials insights
* **Intel optimization**: Optimized for i7-12700K systems

Performance Targets
------------------

* Mechanical Properties: R² ≥ 0.85
* Ballistic Properties: R² ≥ 0.80
* Training Time: 8-12 hours (complete pipeline)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`