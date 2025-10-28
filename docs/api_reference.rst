API Reference
=============

This section provides detailed API documentation for all modules in the Ceramic Armor ML Pipeline.

Data Collection
---------------

.. automodule:: src.data_collection.materials_project_collector
   :members:

.. automodule:: data.data_collection.aflow_collector
   :members:

.. automodule:: data.data_collection.jarvis_collector
   :members:

.. automodule:: data.data_collection.nist_downloader
   :members:

.. automodule:: data.data_collection.data_integrator
   :members:

.. automodule:: data.data_collection.literature_miner
   :members:

Preprocessing
-------------

.. automodule:: src.preprocessing.data_cleaner
   :members:

.. automodule:: src.preprocessing.unit_standardizer
   :members:

.. automodule:: src.preprocessing.outlier_detector
   :members:

.. automodule:: src.preprocessing.missing_value_handler
   :members:

Feature Engineering
-------------------

.. automodule:: src.feature_engineering.derived_properties
   :members:

.. automodule:: src.feature_engineering.compositional_features
   :members:

.. automodule:: src.feature_engineering.microstructure_features
   :members:

.. automodule:: src.feature_engineering.phase_stability
   :members:

Models
------

.. automodule:: src.models.base_model
   :members:

.. automodule:: src.models.xgboost_model
   :members:

.. automodule:: src.models.catboost_model
   :members:

.. automodule:: src.models.random_forest_model
   :members:

.. automodule:: src.models.gradient_boosting_model
   :members:

.. automodule:: src.models.ensemble_model
   :members:

.. automodule:: src.models.transfer_learning
   :members:

Training
--------

.. automodule:: src.training.trainer
   :members:

.. automodule:: src.training.cross_validator
   :members:

.. automodule:: src.training.hyperparameter_tuner
   :members:

Evaluation
----------

.. automodule:: src.evaluation.metrics
   :members:

.. automodule:: src.evaluation.error_analyzer
   :members:

Interpretation
--------------

.. automodule:: src.interpretation.shap_analyzer
   :members:

.. automodule:: src.interpretation.visualization
   :members:

.. automodule:: src.interpretation.materials_insights
   :members:

Utilities
---------

.. automodule:: src.utils.intel_optimizer
   :members: