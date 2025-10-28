#!/usr/bin/env python3
"""
Data Quality Inspection Script for Ceramic Armor ML Pipeline.

Loads existing data at each stage (raw → processed → features), generates statistics
using existing data processing pipeline, creates data quality report, and flags
issues for manual review in existing workflow.

Requirements: 3.3 - Data quality inspection for existing system
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.config_loader import load_config
    from src.utils.data_utils import safe_load_data, validate_data_schema
    from src.preprocessing.data_cleaner import DataCleaner
    from src.preprocessing.outlier_detector import OutlierDetector
    from src.feature_engineering.compositional_features import CompositionalFeatures
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class DataQualityInspector:
    """Comprehensive data quality inspection for the ML pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.quality_issues = []
        self.statistics = {}
        self.recommendations = []
        
        # Set up plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load project configuration."""
        try:
            config_path = self.project_root / 'config' / 'config.yaml'
            return load_config(str(config_path))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def inspect_raw_data(self) -> Dict[str, Any]:
        """Inspect raw data quality from all sources."""
        logger.info("🔍 Inspecting Raw Data Quality")
        
        raw_data_stats = {}
        data_sources = ['materials_project', 'aflow', 'jarvis', 'nist']
        
        for source in data_sources:
            logger.info(f"  Checking {source} data...")
            
            # Look for data files
            source_dir = self.project_root / 'data' / 'raw' / source
            if not source_dir.exists():
                logger.warning(f"    ⚠️  No {source} directory found")
                continue
            
            # Find CSV files
            csv_files = list(source_dir.glob('*.csv'))
            if not csv_files:
                logger.warning(f"    ⚠️  No CSV files found in {source}")
                continue
            
            source_stats = {
                'files_found': len(csv_files),
                'total_materials': 0,
                'ceramic_systems': set(),
                'columns': set(),
                'null_percentages': {},
                'data_types': {},
                'issues': []
            }
            
            for csv_file in csv_files:
                try:
                    data = pd.read_csv(csv_file)
                    
                    if len(data) == 0:
                        source_stats['issues'].append(f"Empty file: {csv_file.name}")
                        continue
                    
                    source_stats['total_materials'] += len(data)
                    source_stats['columns'].update(data.columns)
                    
                    # Identify ceramic systems
                    if 'formula' in data.columns:
                        formulas = data['formula'].dropna().unique()
                        for formula in formulas:
                            if any(ceramic in str(formula) for ceramic in ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']):
                                source_stats['ceramic_systems'].add(formula)
                    
                    # Calculate null percentages
                    null_pct = (data.isnull().sum() / len(data) * 100).to_dict()
                    for col, pct in null_pct.items():
                        if col not in source_stats['null_percentages']:
                            source_stats['null_percentages'][col] = []
                        source_stats['null_percentages'][col].append(pct)
                    
                    # Data types
                    for col, dtype in data.dtypes.items():
                        source_stats['data_types'][col] = str(dtype)
                    
                    logger.info(f"    ✅ {csv_file.name}: {len(data)} materials")
                    
                except Exception as e:
                    source_stats['issues'].append(f"Error reading {csv_file.name}: {e}")
                    logger.error(f"    ❌ Error reading {csv_file.name}: {e}")
            
            # Convert sets to lists for serialization
            source_stats['ceramic_systems'] = list(source_stats['ceramic_systems'])
            source_stats['columns'] = list(source_stats['columns'])
            
            # Average null percentages
            for col, pcts in source_stats['null_percentages'].items():
                source_stats['null_percentages'][col] = np.mean(pcts)
            
            raw_data_stats[source] = source_stats
            
            if source_stats['total_materials'] > 0:
                logger.info(f"  ✅ {source}: {source_stats['total_materials']} materials, "
                          f"{len(source_stats['ceramic_systems'])} ceramic systems")
            else:
                logger.warning(f"  ⚠️  {source}: No materials found")
        
        self.statistics['raw_data'] = raw_data_stats
        return raw_data_stats
    
    def inspect_processed_data(self) -> Dict[str, Any]:
        """Inspect processed data quality after cleaning and standardization."""
        logger.info("🔧 Inspecting Processed Data Quality")
        
        processed_dir = self.project_root / 'data' / 'processed'
        if not processed_dir.exists():
            logger.warning("  ⚠️  No processed data directory found")
            return {}
        
        processed_stats = {
            'files_found': 0,
            'total_materials': 0,
            'ceramic_systems': {},
            'property_coverage': {},
            'data_quality_metrics': {},
            'issues': []
        }
        
        # Look for processed CSV files
        csv_files = list(processed_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("  ⚠️  No processed CSV files found")
            return processed_stats
        
        processed_stats['files_found'] = len(csv_files)
        
        for csv_file in csv_files:
            try:
                data = pd.read_csv(csv_file)
                
                if len(data) == 0:
                    processed_stats['issues'].append(f"Empty processed file: {csv_file.name}")
                    continue
                
                processed_stats['total_materials'] += len(data)
                
                # Analyze ceramic systems
                if 'formula' in data.columns:
                    system_counts = {}
                    for formula in data['formula'].dropna():
                        for ceramic in ['SiC', 'Al2O3', 'B4C', 'WC', 'TiC']:
                            if ceramic in str(formula):
                                system_counts[ceramic] = system_counts.get(ceramic, 0) + 1
                                break
                    processed_stats['ceramic_systems'].update(system_counts)
                
                # Analyze property coverage
                target_properties = [
                    'youngs_modulus', 'bulk_modulus', 'shear_modulus', 'poisson_ratio',
                    'compressive_strength', 'tensile_strength', 'vickers_hardness',
                    'fracture_toughness_mode_i', 'thermal_conductivity', 'density'
                ]
                
                for prop in target_properties:
                    if prop in data.columns:
                        non_null_count = data[prop].notna().sum()
                        coverage_pct = (non_null_count / len(data)) * 100
                        processed_stats['property_coverage'][prop] = coverage_pct
                
                # Data quality metrics
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if col not in processed_stats['data_quality_metrics']:
                        processed_stats['data_quality_metrics'][col] = {}
                    
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        processed_stats['data_quality_metrics'][col] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'null_percentage': float((data[col].isnull().sum() / len(data)) * 100),
                            'outlier_percentage': self._calculate_outlier_percentage(col_data)
                        }
                
                logger.info(f"  ✅ {csv_file.name}: {len(data)} materials")
                
            except Exception as e:
                processed_stats['issues'].append(f"Error reading {csv_file.name}: {e}")
                logger.error(f"  ❌ Error reading {csv_file.name}: {e}")
        
        # Check for data quality issues
        self._check_processed_data_issues(processed_stats)
        
        self.statistics['processed_data'] = processed_stats
        return processed_stats
    
    def inspect_feature_data(self) -> Dict[str, Any]:
        """Inspect engineered features quality."""
        logger.info("⚙️  Inspecting Feature Data Quality")
        
        features_dir = self.project_root / 'data' / 'features'
        if not features_dir.exists():
            logger.warning("  ⚠️  No features data directory found")
            return {}
        
        feature_stats = {
            'files_found': 0,
            'total_materials': 0,
            'feature_count': 0,
            'feature_categories': {},
            'feature_quality': {},
            'correlation_issues': [],
            'issues': []
        }
        
        # Look for feature CSV files
        csv_files = list(features_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("  ⚠️  No feature CSV files found")
            return feature_stats
        
        feature_stats['files_found'] = len(csv_files)
        
        for csv_file in csv_files:
            try:
                data = pd.read_csv(csv_file)
                
                if len(data) == 0:
                    feature_stats['issues'].append(f"Empty feature file: {csv_file.name}")
                    continue
                
                feature_stats['total_materials'] += len(data)
                
                # Count features (exclude ID columns)
                feature_columns = [col for col in data.columns 
                                 if col not in ['material_id', 'formula', 'ceramic_system']]
                feature_stats['feature_count'] = len(feature_columns)
                
                # Categorize features
                feature_categories = {
                    'compositional': [],
                    'structural': [],
                    'derived': [],
                    'thermal': [],
                    'mechanical': [],
                    'electronic': []
                }
                
                for col in feature_columns:
                    col_lower = col.lower()
                    if any(term in col_lower for term in ['atomic', 'element', 'composition']):
                        feature_categories['compositional'].append(col)
                    elif any(term in col_lower for term in ['crystal', 'structure', 'lattice']):
                        feature_categories['structural'].append(col)
                    elif any(term in col_lower for term in ['thermal', 'temperature', 'heat']):
                        feature_categories['thermal'].append(col)
                    elif any(term in col_lower for term in ['modulus', 'strength', 'hardness']):
                        feature_categories['mechanical'].append(col)
                    elif any(term in col_lower for term in ['band_gap', 'electronic']):
                        feature_categories['electronic'].append(col)
                    else:
                        feature_categories['derived'].append(col)
                
                feature_stats['feature_categories'] = {k: len(v) for k, v in feature_categories.items()}
                
                # Feature quality analysis
                numeric_features = data[feature_columns].select_dtypes(include=[np.number])
                
                for col in numeric_features.columns:
                    col_data = numeric_features[col].dropna()
                    if len(col_data) > 0:
                        feature_stats['feature_quality'][col] = {
                            'null_percentage': float((numeric_features[col].isnull().sum() / len(data)) * 100),
                            'zero_percentage': float((col_data == 0).sum() / len(col_data) * 100),
                            'infinite_count': int(np.isinf(col_data).sum()),
                            'variance': float(col_data.var()),
                            'range': float(col_data.max() - col_data.min())
                        }
                
                # Check for high correlations
                if len(numeric_features.columns) > 1:
                    corr_matrix = numeric_features.corr().abs()
                    high_corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.95:
                                high_corr_pairs.append((
                                    corr_matrix.columns[i], 
                                    corr_matrix.columns[j], 
                                    corr_matrix.iloc[i, j]
                                ))
                    
                    feature_stats['correlation_issues'] = high_corr_pairs
                
                logger.info(f"  ✅ {csv_file.name}: {len(data)} materials, {len(feature_columns)} features")
                
            except Exception as e:
                feature_stats['issues'].append(f"Error reading {csv_file.name}: {e}")
                logger.error(f"  ❌ Error reading {csv_file.name}: {e}")
        
        # Check for feature engineering issues
        self._check_feature_issues(feature_stats)
        
        self.statistics['feature_data'] = feature_stats
        return feature_stats
    
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method."""
        if len(data) < 4:
            return 0.0
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return (len(outliers) / len(data)) * 100
    
    def _check_processed_data_issues(self, stats: Dict[str, Any]) -> None:
        """Check for issues in processed data."""
        # Check property coverage
        for prop, coverage in stats.get('property_coverage', {}).items():
            if coverage < 50:
                issue = f"Low coverage for {prop}: {coverage:.1f}%"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
        
        # Check ceramic system balance
        systems = stats.get('ceramic_systems', {})
        if systems:
            max_count = max(systems.values())
            min_count = min(systems.values())
            
            if max_count > min_count * 5:  # Imbalance threshold
                issue = f"Imbalanced ceramic systems: {systems}"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
        
        # Check for high null percentages
        for col, metrics in stats.get('data_quality_metrics', {}).items():
            null_pct = metrics.get('null_percentage', 0)
            if null_pct > 30:
                issue = f"High null percentage in {col}: {null_pct:.1f}%"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
    
    def _check_feature_issues(self, stats: Dict[str, Any]) -> None:
        """Check for issues in feature data."""
        # Check for features with high null percentages
        for feature, quality in stats.get('feature_quality', {}).items():
            null_pct = quality.get('null_percentage', 0)
            if null_pct > 20:
                issue = f"High null percentage in feature {feature}: {null_pct:.1f}%"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
            
            # Check for zero variance features
            variance = quality.get('variance', 1)
            if variance < 1e-10:
                issue = f"Zero variance feature: {feature}"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
            
            # Check for infinite values
            inf_count = quality.get('infinite_count', 0)
            if inf_count > 0:
                issue = f"Infinite values in feature {feature}: {inf_count}"
                self.quality_issues.append(issue)
                logger.warning(f"    ⚠️  {issue}")
        
        # Check for highly correlated features
        corr_issues = stats.get('correlation_issues', [])
        if len(corr_issues) > 10:
            issue = f"Many highly correlated features: {len(corr_issues)} pairs"
            self.quality_issues.append(issue)
            logger.warning(f"    ⚠️  {issue}")
    
    def create_data_quality_visualizations(self) -> None:
        """Create visualizations for data quality report."""
        logger.info("📊 Creating Data Quality Visualizations")
        
        figures_dir = self.project_root / 'results' / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Data availability by source
        if 'raw_data' in self.statistics:
            self._plot_data_availability()
        
        # 2. Property coverage heatmap
        if 'processed_data' in self.statistics:
            self._plot_property_coverage()
        
        # 3. Feature quality metrics
        if 'feature_data' in self.statistics:
            self._plot_feature_quality()
        
        logger.info(f"  ✅ Visualizations saved to {figures_dir}")
    
    def _plot_data_availability(self) -> None:
        """Plot data availability by source."""
        raw_stats = self.statistics['raw_data']
        
        sources = []
        material_counts = []
        
        for source, stats in raw_stats.items():
            if stats['total_materials'] > 0:
                sources.append(source.replace('_', ' ').title())
                material_counts.append(stats['total_materials'])
        
        if not sources:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sources, material_counts, color='skyblue', alpha=0.7)
        plt.title('Data Availability by Source', fontsize=14, fontweight='bold')
        plt.xlabel('Data Source')
        plt.ylabel('Number of Materials')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, material_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(material_counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.project_root / 'results' / 'figures' / 'data_availability.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_property_coverage(self) -> None:
        """Plot property coverage heatmap."""
        if 'property_coverage' not in self.statistics['processed_data']:
            return
        
        coverage = self.statistics['processed_data']['property_coverage']
        
        if not coverage:
            return
        
        properties = list(coverage.keys())
        coverages = list(coverage.values())
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap data
        data = np.array(coverages).reshape(1, -1)
        
        sns.heatmap(data, 
                   xticklabels=[prop.replace('_', ' ').title() for prop in properties],
                   yticklabels=['Coverage %'],
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   vmin=0, 
                   vmax=100,
                   cbar_kws={'label': 'Coverage Percentage'})
        
        plt.title('Property Coverage in Processed Data', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.project_root / 'results' / 'figures' / 'property_coverage.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_quality(self) -> None:
        """Plot feature quality metrics."""
        if 'feature_quality' not in self.statistics['feature_data']:
            return
        
        quality_data = self.statistics['feature_data']['feature_quality']
        
        if not quality_data:
            return
        
        # Extract null percentages
        features = list(quality_data.keys())[:20]  # Top 20 features
        null_pcts = [quality_data[f]['null_percentage'] for f in features]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), null_pcts, color='coral', alpha=0.7)
        plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Null Percentage (%)')
        plt.title('Feature Quality: Null Percentages', fontsize=14, fontweight='bold')
        
        # Add threshold line
        plt.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20% Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.project_root / 'results' / 'figures' / 'feature_quality.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on data quality analysis."""
        recommendations = []
        
        # Raw data recommendations
        if 'raw_data' in self.statistics:
            raw_stats = self.statistics['raw_data']
            
            # Check for missing data sources
            expected_sources = ['materials_project', 'aflow', 'jarvis', 'nist']
            missing_sources = [s for s in expected_sources 
                             if s not in raw_stats or raw_stats[s]['total_materials'] == 0]
            
            if missing_sources:
                recommendations.append(f"Collect data from missing sources: {', '.join(missing_sources)}")
        
        # Processed data recommendations
        if 'processed_data' in self.statistics:
            processed_stats = self.statistics['processed_data']
            
            # Check ceramic system balance
            systems = processed_stats.get('ceramic_systems', {})
            if systems:
                min_count = min(systems.values())
                if min_count < 100:
                    recommendations.append("Increase data collection for underrepresented ceramic systems")
            
            # Check property coverage
            coverage = processed_stats.get('property_coverage', {})
            low_coverage_props = [prop for prop, cov in coverage.items() if cov < 50]
            if low_coverage_props:
                recommendations.append(f"Improve data collection for properties: {', '.join(low_coverage_props)}")
        
        # Feature data recommendations
        if 'feature_data' in self.statistics:
            feature_stats = self.statistics['feature_data']
            
            # Check feature count
            if feature_stats.get('feature_count', 0) < 50:
                recommendations.append("Consider engineering additional features for better model performance")
            
            # Check correlation issues
            if len(feature_stats.get('correlation_issues', [])) > 10:
                recommendations.append("Remove highly correlated features to reduce multicollinearity")
        
        # General recommendations based on issues
        if len(self.quality_issues) > 10:
            recommendations.append("Address data quality issues before training models")
        
        self.recommendations = recommendations
        return recommendations
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        logger.info("📋 Generating Data Quality Report")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_quality_issues': len(self.quality_issues),
                'data_stages_inspected': len(self.statistics),
                'recommendations_count': len(self.recommendations)
            },
            'statistics': self.statistics,
            'quality_issues': self.quality_issues,
            'recommendations': self.recommendations,
            'data_readiness_score': self._calculate_readiness_score()
        }
        
        # Save report
        report_path = self.project_root / 'logs' / 'data_quality_report.yaml'
        try:
            with open(report_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Quality report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
        
        return report
    
    def _calculate_readiness_score(self) -> float:
        """Calculate overall data readiness score (0-100)."""
        score = 100.0
        
        # Deduct points for quality issues
        score -= min(len(self.quality_issues) * 5, 50)
        
        # Check data availability
        if 'raw_data' in self.statistics:
            raw_stats = self.statistics['raw_data']
            available_sources = sum(1 for stats in raw_stats.values() 
                                  if stats['total_materials'] > 0)
            if available_sources < 2:
                score -= 20
        
        # Check processed data
        if 'processed_data' in self.statistics:
            processed_stats = self.statistics['processed_data']
            if processed_stats.get('total_materials', 0) < 1000:
                score -= 15
        
        # Check features
        if 'feature_data' in self.statistics:
            feature_stats = self.statistics['feature_data']
            if feature_stats.get('feature_count', 0) < 30:
                score -= 10
        
        return max(score, 0.0)
    
    def run_inspection(self) -> bool:
        """Run complete data quality inspection."""
        logger.info("🚀 Data Quality Inspection Suite")
        logger.info("=" * 60)
        
        inspection_steps = [
            ("Raw Data", self.inspect_raw_data),
            ("Processed Data", self.inspect_processed_data),
            ("Feature Data", self.inspect_feature_data)
        ]
        
        for step_name, step_func in inspection_steps:
            logger.info(f"\n📋 {step_name}")
            logger.info("-" * 40)
            
            try:
                step_func()
            except Exception as e:
                logger.error(f"❌ {step_name} inspection failed: {e}")
                self.quality_issues.append(f"{step_name} inspection failed: {e}")
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Create visualizations
        try:
            self.create_data_quality_visualizations()
        except Exception as e:
            logger.warning(f"⚠️  Visualization creation failed: {e}")
        
        # Generate report
        report = self.generate_quality_report()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY INSPECTION SUMMARY")
        logger.info("=" * 60)
        
        readiness_score = report['data_readiness_score']
        logger.info(f"Data Readiness Score: {readiness_score:.1f}/100")
        
        if readiness_score >= 80:
            logger.info("🎉 Data quality is excellent! Ready for model training.")
            status = "EXCELLENT"
        elif readiness_score >= 60:
            logger.info("✅ Data quality is good. Minor improvements recommended.")
            status = "GOOD"
        elif readiness_score >= 40:
            logger.warning("⚠️  Data quality needs improvement before training.")
            status = "NEEDS_IMPROVEMENT"
        else:
            logger.error("❌ Data quality is poor. Major fixes required.")
            status = "POOR"
        
        logger.info(f"Quality Issues Found: {len(self.quality_issues)}")
        logger.info(f"Recommendations: {len(self.recommendations)}")
        
        if self.quality_issues:
            logger.info("\n🔍 Top Quality Issues:")
            for issue in self.quality_issues[:5]:
                logger.info(f"  • {issue}")
        
        if self.recommendations:
            logger.info("\n💡 Key Recommendations:")
            for rec in self.recommendations[:3]:
                logger.info(f"  • {rec}")
        
        logger.info(f"\n📊 Full report: logs/data_quality_report.yaml")
        logger.info(f"📈 Visualizations: results/figures/")
        
        return status in ["EXCELLENT", "GOOD"]


def main():
    """Main entry point."""
    inspector = DataQualityInspector()
    success = inspector.run_inspection()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())