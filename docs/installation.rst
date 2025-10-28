Installation Guide
==================

System Requirements
-------------------

* Python 3.11+
* Intel i7-12700K (recommended) or equivalent
* 32GB RAM (recommended)
* 100GB free disk space
* Internet connection for data collection

Environment Setup
-----------------

1. **Create Conda Environment**

.. code-block:: bash

   conda create -n ceramic_ml python=3.11
   conda activate ceramic_ml

2. **Install Dependencies**

.. code-block:: bash

   pip install -r requirements.txt
   pip install pymatgen jarvis-tools requests beautifulsoup4

3. **Verify Installation**

.. code-block:: python

   import xgboost
   import catboost
   import pymatgen
   from jarvis.db.figshare import data
   print("✓ All dependencies installed successfully")

API Configuration
-----------------

Create ``config/api_keys.yaml``:

.. code-block:: yaml

   materials_project: "YOUR_MP_API_KEY"
   semantic_scholar: "YOUR_SS_API_KEY"  # Optional

**Getting API Keys:**

* **Materials Project**: Register at https://materialsproject.org/api
* **Semantic Scholar**: Optional, get from https://www.semanticscholar.org/product/api

Data Directory Setup
--------------------

Create required directories:

.. code-block:: bash

   mkdir -p data/raw/{materials_project,aflow,jarvis,nist,literature}
   mkdir -p data/processed
   mkdir -p data/features
   mkdir -p results/{models,predictions,figures,metrics}

NIST Data Placement
-------------------

Place NIST CSV files in system-specific directories:

.. code-block:: bash

   data/raw/nist/
   ├── sic/
   │   └── sic_properties.csv
   ├── al2o3/
   │   └── alumina_data.csv
   ├── b4c/
   │   └── boron_carbide.csv
   ├── wc/
   │   └── tungsten_carbide.csv
   └── tic/
       └── titanium_carbide.csv

**Required CSV Format:**

.. code-block:: csv

   formula,density,youngs_modulus,vickers_hardness,fracture_toughness
   SiC,3.21,410,28,4.6
   Al2O3,3.95,370,15,4.2

Verification
------------

Run the verification script:

.. code-block:: bash

   python -c "
   from src.preprocessing.unit_standardizer import PRESSURE_TO_GPA
   from data.data_collection.aflow_collector import AFLOWCollector
   from data.data_collection.jarvis_collector import JARVISCollector
   print('✓ Unit standardization ready:', list(PRESSURE_TO_GPA.keys()))
   print('✓ AFLOW collector ready')
   print('✓ JARVIS collector ready')
   print('✓ Installation complete!')
   "