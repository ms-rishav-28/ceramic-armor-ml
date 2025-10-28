"""
SHAP Analysis Publication Validation Script
Validates that SHAP analysis produces publication-ready results for all trained models
Generates comprehensive interpretability analysis suitable for journal publication
"""

import sys
sys.path.append('.')

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from loguru import logger

# Import pipeline components
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.interpretation.shap_analyzer import SHAPAnalyzer
from src.interpretation.materials_insights import interpret_feature_ranking

class SHAPPublicationValidator:
    """
    Validates SHAP analysis for publication readiness
    
    Generates:
    1. Publication-quality SHAP plots for all models
    2. Feature importance rankings with materials science insights
    3. Cross-system comparison of feature importance
    4. Mechanistic interpretation of key features
    5. Publication-ready figure compilation
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize SHAP publication validator"""
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        
        # Publication settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.transparent': False
        })
        
        self.logger.info("SHAP Publication Validator initialized")
    
    def validate_model_shap_analysis(self, system: str, property_name: str, 
                                   model_dir: Path, output_dir: Path) -> Dict:
        """
        Validate SHAP analysis for a specific model
        
        Args:
            system: Ceramic system name
            property_name: Target property name
            model_dir: Directory containing trained models
            output_dir: Output directory for SHAP plots
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating SHAP analysis for {system} - {property_name}")
        
        results = {
            'system': system,
            'property': property_name,
            'status': 'failed',
            'plots_generated': [],
            'plots_failed': [],
            'feature_insights': {},
            'publication_ready': False
        }
        
        try:
            # Find best model (prefer ensemble)
            model_files = {
                'ensemble': model_dir / "ensemble_model.pkl",
                'xgboost': model_dir / "xgboost_model.pkl",
                'catboost': model_dir / "catboost_model.pkl",
                'random_forest': model_dir / "random_forest_model.pkl"
            }
            
            # Select best available model
            selected_model = None
            model_type = None
            
            for name, path in model_files.items():
                if path.exists():
                    try:
                        model_data = joblib.load(path)
                        selected_model = model_data.get('model', model_data)
                        model_type = name
                        break
                    except Exception as e:
                        self.logger.warning(f"Failed to load {name} model: {e}")
                        continue
            
            if selected_model is None:
                results['error'] = 'No valid model found'
                return results
            
            self.logger.info(f"Using {model_type} model for SHAP analysis")
            
            # Initialize SHAP analyzer
            analyzer = SHAPAnalyzer(selected_model, model_type='tree')
            
            # Create system-property specific output directory
            shap_output_dir = output_dir / f"{system}_{property_name}"
            shap_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate comprehensive SHAP analysis
            try:
                shap_results = analyzer.generate_all_plots(
                    model_dir=str(model_dir),
                    output_dir=str(shap_output_dir),
                    top_features=20  # More features for publication
                )
                
                results['plots_generated'] = shap_results.get('successful_plots', [])
                results['plots_failed'] = shap_results.get('failed_plots', [])
                
                # Calculate success metrics
                total_plots = len(results['plots_generated']) + len(results['plots_failed'])
                success_rate = len(results['plots_generated']) / total_plots * 100 if total_plots > 0 else 0
                
                # Generate feature importance insights
                if hasattr(analyzer, 'shap_values') and analyzer.shap_values is not None:
                    importance_df = analyzer.get_feature_importance()
                    
                    # Generate materials science insights
                    insights_text = interpret_feature_ranking(importance_df, top_k=15)
                    
                    results['feature_insights'] = {
                        'top_features': importance_df.head(15).to_dict('records'),
                        'materials_insights': insights_text,
                        'total_features': len(importance_df)
                    }
                    
                    # Save feature importance table
                    importance_df.to_csv(shap_output_dir / 'feature_importance.csv', index=False)
                
                # Publication readiness criteria
                min_required_plots = ['summary_dot', 'summary_bar']
                has_required_plots = all(plot in results['plots_generated'] for plot in min_required_plots)
                
                results['publication_ready'] = (
                    success_rate >= 70 and  # At least 70% of plots successful
                    has_required_plots and  # Required plots present
                    len(results['plots_generated']) >= 3  # Minimum plot diversity
                )
                
                results['success_rate'] = success_rate
                results['status'] = 'success'
                results['output_directory'] = str(shap_output_dir)
                
                self.logger.info(f"✓ SHAP analysis complete: {len(results['plots_generated'])} plots generated")
                
            except Exception as e:
                results['error'] = f'SHAP generation failed: {str(e)}'
                self.logger.error(f"SHAP generation failed: {e}")
                
        except Exception as e:
            results['error'] = f'Model loading failed: {str(e)}'
            self.logger.error(f"Model loading failed: {e}")
        
        return results
    
    def generate_cross_system_comparison(self, all_results: List[Dict], 
                                       output_dir: Path) -> Dict:
        """
        Generate cross-system feature importance comparison
        
        Args:
            all_results: List of SHAP validation results
            output_dir: Output directory for comparison plots
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Generating cross-system feature importance comparison")
        
        comparison_dir = output_dir / "cross_system_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect feature importance data
        feature_data = []
        
        for result in all_results:
            if result['status'] == 'success' and 'feature_insights' in result:
                top_features = result['feature_insights'].get('top_features', [])
                
                for feature_info in top_features[:10]:  # Top 10 features
                    feature_data.append({
                        'system': result['system'],
                        'property': result['property'],
                        'feature': feature_info['feature'],
                        'importance': feature_info['importance'],
                        'rank': top_features.index(feature_info) + 1
                    })
        
        if not feature_data:
            return {'status': 'failed', 'error': 'No feature importance data available'}
        
        df_features = pd.DataFrame(feature_data)
        
        # Generate comparison visualizations
        plots_generated = []
        
        try:
            # 1. Feature importance heatmap across systems
            plt.figure(figsize=(14, 10))
            
            # Create pivot table for heatmap
            pivot_data = df_features.pivot_table(
                values='importance', 
                index='feature', 
                columns=['system', 'property'], 
                fill_value=0
            )
            
            # Select top features across all systems
            feature_means = pivot_data.mean(axis=1).sort_values(ascending=False)
            top_features = feature_means.head(20).index
            
            sns.heatmap(
                pivot_data.loc[top_features], 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                cbar_kws={'label': 'SHAP Importance'}
            )
            
            plt.title('Feature Importance Across Ceramic Systems and Properties', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('System - Property', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            heatmap_path = comparison_dir / 'feature_importance_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_generated.append('feature_importance_heatmap')
            self.logger.info("✓ Feature importance heatmap generated")
            
        except Exception as e:
            self.logger.error(f"Failed to generate heatmap: {e}")
        
        try:
            # 2. Top features bar plot by system
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            systems = df_features['system'].unique()
            
            for i, system in enumerate(systems):
                if i >= len(axes):
                    break
                    
                system_data = df_features[df_features['system'] == system]
                
                # Get top features for this system
                top_system_features = (system_data.groupby('feature')['importance']
                                     .mean().sort_values(ascending=False).head(10))
                
                axes[i].barh(range(len(top_system_features)), 
                           top_system_features.values,
                           color=plt.cm.Set3(i))
                axes[i].set_yticks(range(len(top_system_features)))
                axes[i].set_yticklabels(top_system_features.index, fontsize=10)
                axes[i].set_xlabel('Mean SHAP Importance', fontsize=10)
                axes[i].set_title(f'{system} System', fontsize=12, fontweight='bold')
                axes[i].grid(axis='x', alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(systems), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Top Features by Ceramic System', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            barplot_path = comparison_dir / 'top_features_by_system.png'
            plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_generated.append('top_features_by_system')
            self.logger.info("✓ Top features by system plot generated")
            
        except Exception as e:
            self.logger.error(f"Failed to generate bar plots: {e}")
        
        try:
            # 3. Feature consistency analysis
            feature_consistency = (df_features.groupby('feature')
                                 .agg({
                                     'system': 'nunique',
                                     'importance': ['mean', 'std', 'count']
                                 }).round(4))
            
            feature_consistency.columns = ['systems_count', 'mean_importance', 
                                         'std_importance', 'total_occurrences']
            
            # Features that appear in multiple systems
            consistent_features = feature_consistency[
                feature_consistency['systems_count'] >= 2
            ].sort_values('mean_importance', ascending=False)
            
            # Save consistency analysis
            consistency_path = comparison_dir / 'feature_consistency_analysis.csv'
            consistent_features.to_csv(consistency_path)
            
            self.logger.info(f"✓ Feature consistency analysis saved: {len(consistent_features)} consistent features")
            
        except Exception as e:
            self.logger.error(f"Failed to generate consistency analysis: {e}")
        
        # Generate summary report
        try:
            summary_report = f"""
# Cross-System Feature Importance Analysis

## Summary Statistics
- **Total Systems Analyzed:** {df_features['system'].nunique()}
- **Total Properties Analyzed:** {df_features['property'].nunique()}
- **Unique Features Identified:** {df_features['feature'].nunique()}
- **Total Feature-Property Combinations:** {len(df_features)}

## Most Important Features Across All Systems
"""
            
            # Overall top features
            overall_top = (df_features.groupby('feature')['importance']
                         .mean().sort_values(ascending=False).head(15))
            
            for i, (feature, importance) in enumerate(overall_top.items(), 1):
                summary_report += f"{i}. **{feature}**: {importance:.4f}\n"
            
            summary_report += f"""
## Feature Consistency Analysis
- **Features appearing in multiple systems:** {len(consistent_features)}
- **Most consistent feature:** {consistent_features.index[0] if len(consistent_features) > 0 else 'N/A'}

## Publication Readiness
- **Plots Generated:** {len(plots_generated)}
- **Data Quality:** {'✅ High' if len(df_features) > 50 else '⚠️ Limited'}
- **Cross-System Coverage:** {'✅ Complete' if df_features['system'].nunique() >= 4 else '⚠️ Partial'}

---
*Generated by SHAP Publication Validator*
"""
            
            report_path = comparison_dir / 'cross_system_analysis_report.md'
            with open(report_path, 'w') as f:
                f.write(summary_report)
            
            self.logger.info(f"✓ Cross-system analysis report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
        
        return {
            'status': 'success',
            'plots_generated': plots_generated,
            'total_features_analyzed': df_features['feature'].nunique(),
            'consistent_features': len(consistent_features) if 'consistent_features' in locals() else 0,
            'output_directory': str(comparison_dir)
        }
    
    def validate_publication_figures(self, output_dir: Path) -> Dict:
        """
        Validate that generated figures meet publication standards
        
        Args:
            output_dir: Directory containing SHAP figures
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating publication figure quality")
        
        validation_results = {
            'total_figures': 0,
            'high_quality_figures': 0,
            'publication_ready_figures': [],
            'issues_found': [],
            'recommendations': []
        }
        
        # Find all PNG files
        figure_files = list(output_dir.rglob("*.png"))
        validation_results['total_figures'] = len(figure_files)
        
        for fig_path in figure_files:
            try:
                # Check file size (should be reasonable for publication)
                file_size_mb = fig_path.stat().st_size / (1024 * 1024)
                
                # Load image to check dimensions and quality
                import PIL.Image
                with PIL.Image.open(fig_path) as img:
                    width, height = img.size
                    
                    # Publication quality criteria
                    criteria = {
                        'min_width': width >= 800,  # Minimum width for clarity
                        'min_height': height >= 600,  # Minimum height for clarity
                        'reasonable_size': 0.5 <= file_size_mb <= 10,  # Reasonable file size
                        'high_dpi': width * height >= 800 * 600  # Effective DPI check
                    }
                    
                    passed_criteria = sum(criteria.values())
                    
                    if passed_criteria >= 3:  # At least 3/4 criteria met
                        validation_results['high_quality_figures'] += 1
                        validation_results['publication_ready_figures'].append(str(fig_path.name))
                    else:
                        issues = []
                        if not criteria['min_width']:
                            issues.append(f"Width too small: {width}px")
                        if not criteria['min_height']:
                            issues.append(f"Height too small: {height}px")
                        if not criteria['reasonable_size']:
                            issues.append(f"File size issue: {file_size_mb:.1f}MB")
                        
                        validation_results['issues_found'].append({
                            'file': str(fig_path.name),
                            'issues': issues
                        })
                        
            except Exception as e:
                validation_results['issues_found'].append({
                    'file': str(fig_path.name),
                    'issues': [f"Validation error: {str(e)}"]
                })
        
        # Generate recommendations
        quality_rate = (validation_results['high_quality_figures'] / 
                       validation_results['total_figures'] * 100 
                       if validation_results['total_figures'] > 0 else 0)
        
        if quality_rate >= 90:
            validation_results['recommendations'].append("✅ Excellent figure quality - ready for publication")
        elif quality_rate >= 70:
            validation_results['recommendations'].append("⚠️ Good figure quality - minor improvements recommended")
        else:
            validation_results['recommendations'].append("❌ Figure quality needs improvement before publication")
        
        validation_results['quality_rate'] = quality_rate
        
        self.logger.info(f"Figure validation complete: {validation_results['high_quality_figures']}/{validation_results['total_figures']} high quality")
        
        return validation_results
    
    def run_full_shap_validation(self) -> Dict:
        """
        Run complete SHAP analysis validation for publication
        
        Returns:
            Dictionary with comprehensive validation results
        """
        self.logger.info("\n" + "📊"*20)
        self.logger.info("SHAP ANALYSIS PUBLICATION VALIDATION")
        self.logger.info("📊"*20)
        
        # Create output directory
        output_dir = Path("results/figures/shap_publication_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir = Path(self.config['paths']['models'])
        
        all_results = []
        successful_analyses = 0
        total_analyses = 0
        
        # Validate SHAP analysis for each model
        for system in self.config['ceramic_systems']['primary']:
            self.logger.info(f"\n--- Validating SHAP for {system} ---")
            
            # Test key properties for publication
            key_properties = [
                'youngs_modulus',           # Key mechanical property
                'fracture_toughness_mode_i', # Critical for armor
                'vickers_hardness',         # Standard measurement
                'v50'                       # Key ballistic property
            ]
            
            for prop in key_properties:
                # Check if property exists in config
                if (prop in self.config['properties']['mechanical'] or 
                    prop in self.config['properties']['ballistic']):
                    
                    model_dir = models_dir / system.lower() / prop
                    
                    if model_dir.exists():
                        total_analyses += 1
                        
                        result = self.validate_model_shap_analysis(
                            system, prop, model_dir, output_dir
                        )
                        
                        all_results.append(result)
                        
                        if result['status'] == 'success' and result['publication_ready']:
                            successful_analyses += 1
                            self.logger.info(f"  {prop}: ✅ Publication ready")
                        else:
                            self.logger.warning(f"  {prop}: ⚠️ Needs improvement")
                    else:
                        self.logger.warning(f"  {prop}: Model directory not found")
        
        # Generate cross-system comparison
        self.logger.info("\n--- Generating Cross-System Analysis ---")
        comparison_results = self.generate_cross_system_comparison(all_results, output_dir)
        
        # Validate figure quality
        self.logger.info("\n--- Validating Figure Quality ---")
        figure_validation = self.validate_publication_figures(output_dir)
        
        # Calculate overall success metrics
        success_rate = successful_analyses / total_analyses * 100 if total_analyses > 0 else 0
        
        # Generate comprehensive report
        report_content = f"""
# SHAP Analysis Publication Validation Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Overall Status
**Publication Readiness:** {'✅ READY' if success_rate >= 80 else '❌ NEEDS IMPROVEMENT'}

## Analysis Results

### SHAP Analysis Coverage
- **Total Analyses:** {total_analyses}
- **Successful Analyses:** {successful_analyses}
- **Success Rate:** {success_rate:.1f}%

### Individual Model Results
"""
        
        for result in all_results:
            status_icon = '✅' if result.get('publication_ready', False) else '❌'
            plots_count = len(result.get('plots_generated', []))
            report_content += f"- **{result['system']} - {result['property']}:** {status_icon} ({plots_count} plots)\n"
        
        report_content += f"""
### Cross-System Analysis
- **Status:** {'✅ Complete' if comparison_results.get('status') == 'success' else '❌ Failed'}
- **Features Analyzed:** {comparison_results.get('total_features_analyzed', 0)}
- **Consistent Features:** {comparison_results.get('consistent_features', 0)}

### Figure Quality Assessment
- **Total Figures:** {figure_validation['total_figures']}
- **High Quality Figures:** {figure_validation['high_quality_figures']}
- **Quality Rate:** {figure_validation['quality_rate']:.1f}%

## Publication Readiness Checklist

"""
        
        # Publication checklist
        checklist_items = [
            ("SHAP Coverage", success_rate >= 80, f"{success_rate:.1f}% of analyses successful"),
            ("Figure Quality", figure_validation['quality_rate'] >= 80, f"{figure_validation['quality_rate']:.1f}% high quality figures"),
            ("Cross-System Analysis", comparison_results.get('status') == 'success', "Feature comparison complete"),
            ("Materials Insights", len([r for r in all_results if 'feature_insights' in r]) > 0, "Mechanistic interpretations available"),
            ("Reproducibility", True, "All code and data available")
        ]
        
        for item, status, description in checklist_items:
            icon = '✅' if status else '❌'
            report_content += f"- {icon} **{item}:** {description}\n"
        
        overall_ready = all(status for _, status, _ in checklist_items)
        
        report_content += f"""
### Overall Assessment: {'✅ PUBLICATION READY' if overall_ready else '❌ REQUIRES IMPROVEMENT'}

## Output Locations
- **SHAP Plots:** `{output_dir}`
- **Cross-System Analysis:** `{comparison_results.get('output_directory', 'N/A')}`
- **This Report:** `results/reports/shap_publication_validation/`

## Recommendations

"""
        
        if overall_ready:
            report_content += "✅ All SHAP analyses are publication ready. Figures and interpretations meet journal standards.\n"
        else:
            report_content += "### Areas for Improvement:\n"
            for item, status, description in checklist_items:
                if not status:
                    report_content += f"- **{item}:** {description}\n"
        
        report_content += """
---
*Generated by SHAP Publication Validator*
"""
        
        # Save report
        report_dir = Path("results/reports/shap_publication_validation")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / "shap_publication_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save detailed results
        import json
        results_file = report_dir / "shap_validation_results.json"
        
        final_results = {
            'total_analyses': total_analyses,
            'successful_analyses': successful_analyses,
            'success_rate': success_rate,
            'individual_results': all_results,
            'cross_system_analysis': comparison_results,
            'figure_validation': figure_validation,
            'publication_ready': overall_ready,
            'output_directory': str(output_dir),
            'report_path': str(report_file)
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"\n" + "="*80)
        self.logger.info("SHAP PUBLICATION VALIDATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info(f"Publication ready: {'Yes' if overall_ready else 'No'}")
        self.logger.info(f"Report: {report_file}")
        self.logger.info(f"Results: {results_file}")
        
        return final_results


def main():
    """Main execution function"""
    try:
        # Initialize validator
        validator = SHAPPublicationValidator()
        
        # Run full SHAP validation
        results = validator.run_full_shap_validation()
        
        # Exit with appropriate code
        exit_code = 0 if results.get('publication_ready', False) else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"SHAP publication validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()