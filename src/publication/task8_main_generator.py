"""
Task 8 Main Generator - Coordinates all publication components
Implements comprehensive publication-grade analysis meeting top-tier journal standards
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configure headless plotting for Windows compatibility
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
import json
import yaml
from datetime import datetime

from .publication_analyzer import PublicationAnalyzer
from .figure_generator import PublicationFigureGenerator
from .manuscript_generator import ManuscriptGenerator
from .task8_mechanistic_analysis import MechanisticAnalysisGenerator
from ..interpretation.comprehensive_interpretability import ComprehensiveInterpretabilityAnalyzer


class Task8MainGenerator:
    """
    Task 8 Main Generator - Coordinates all publication components
    
    Implements all Task 8 requirements:
    - Tree-based model superiority analysis
    - Mechanistic interpretation with literature references
    - Project structure overview
    - Publication-ready figures with statistical significance
    - Scientific documentation with ballistic response factors
    - Journal-standard outputs for Nature Materials, Acta Materialia, Materials & Design
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Task 8 main generator"""
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize component generators
        self.publication_analyzer = PublicationAnalyzer(config_path)
        self.figure_generator = PublicationFigureGenerator()
        sel