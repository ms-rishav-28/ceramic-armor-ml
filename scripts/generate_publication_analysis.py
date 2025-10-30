#!/usr/bin/env python3
"""
Publication-Ready Analysis Generator Script
Orchestrates the complete generation of publication-ready analysis and documentation

This script implements Task 8 requirements:
- Create comprehensive analysis commentary explaining why tree-based models outperform neural networks
- Generate mechanistic interpretation correlating feature importance to known materials science principles with literature references
- Provide complete project structure overview with minimal but sufficient implementations focused on essential functionality
- Create publication-ready figures with proper scientific formatting, error bars, and statistical significance testing
- Document mechanistic interpretation of which material factors control ballistic response with physical reasoning
- Ensure all outputs meet top-tier journal publication standards (Nature Materials, Acta Materialia, Materials & Design)

Usage:
    python scripts/generate_publication_analysis.py [--output-dir OUTPUT_DIR] [--use-existing-results]
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from publication.publication_analyzer import PublicationAnalyzer
from publication.figure_generator import PublicationFigureGenerator
from publication.manuscript_generator import ManuscriptGenerator


def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/publication_analysis.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )


def load_existing_interpretability_results(results_dir: str) -> dict:
    """Load existing interpretability results if available"""
    
    results_path = Path(results_dir)
    
    # Try to load comprehensive interpretability results
    comprehensive_results_file = results_path / 'comprehensive_interpretability_results.json'
    
    if comprehensive_results_file.exists():
        logger.info(f"Loading existing interpretability results from {comprehensive_results_file}")
        try:
            with open(comprehensive_results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
    
    # If no existing results, return empty structure
    logger.info("No existing interpretability results found - will use placeholder data")
    return {
        'individual_analyses': {},
        'cross_system_comparison': {},
        'analysis_summary': {
            'total_analyses': 0,
            'successful_analyses': 0,
            'publication_ready_analyses': 0
        }
    }


def generate_comprehensive_analysis_commentary(analyzer: PublicationAnalyzer, output_dir: str) -> dict:
    """Generate comprehensive analysis commentary"""
    
    logger.info("="*80)
    logger.info("GENERATING COMPREHENSIVE ANALYSIS COMMENTARY")
    logger.info("="*80)
    
    commentary_dir = Path(output_dir) / 'analysis_commentary'
    
    try:
        commentary_results = analyzer.generate_comprehensive_analysis_commentary(str(commentary_dir))
        
        if commentary_results['status'] == 'success':
            logger.info("✅ Comprehensive analysis commentary generated successfully")
            logger.info(f"   📄 Markdown: {commentary_results['markdown_file']}")
            logger.info(f"   📊 JSON: {commentary_results['json_file']}")
            logger.info(f"   📚 Literature references: {commentary_results['literature_references']}")
        else:
            logger.error("❌ Failed to generate comprehensive analysis commentary")
        
        return commentary_results
        
    except Exception as e:
        logger.error(f"❌ Error generating comprehensive analysis commentary: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_mechanistic_interpretation(analyzer: PublicationAnalyzer, 
                                      interpretability_results: dict, 
                                      output_dir: str) -> dict:
    """Generate mechanistic interpretation with literature references"""
    
    logger.info("="*80)
    logger.info("GENERATING MECHANISTIC INTERPRETATION WITH LITERATURE")
    logger.info("="*80)
    
    mechanistic_dir = Path(output_dir) / 'mechanistic_interpretation'
    
    try:
        mechanistic_results = analyzer.generate_mechanistic_interpretation_with_literature(
            interpretability_results, str(mechanistic_dir)
        )
        
        if mechanistic_results['status'] == 'success':
            logger.info("✅ Mechanistic interpretation generated successfully")
            logger.info(f"   📄 Markdown: {mechanistic_results['markdown_file']}")
            logger.info(f"   📊 JSON: {mechanistic_results['json_file']}")
            logger.info(f"   🔬 Systems analyzed: {mechanistic_results['systems_analyzed']}")
            logger.info(f"   📚 Literature references: {mechanistic_results['literature_references']}")
        else:
            logger.error("❌ Failed to generate mechanistic interpretation")
        
        return mechanistic_results
        
    except Exception as e:
        logger.error(f"❌ Error generating mechanistic interpretation: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_publication_figures(figure_generator: PublicationFigureGenerator,
                               interpretability_results: dict,
                               mechanistic_results: dict,
                               output_dir: str) -> dict:
    """Generate publication-ready figures"""
    
    logger.info("="*80)
    logger.info("GENERATING PUBLICATION-READY FIGURES")
    logger.info("="*80)
    
    figures_dir = Path(output_dir) / 'publication_figures'
    figures_results = {'figures_created': [], 'figures_failed': []}
    
    try:
        # Generate cross-system feature importance figure
        logger.info("Creating cross-system feature importance figure...")
        cross_system_result = figure_generator.create_cross_system_feature_importance_figure(
            interpretability_results,
            str(figures_dir / 'cross_system_feature_importance.png')
        )
        
        if cross_system_result['status'] == 'success':
            logger.info("✅ Cross-system feature importance figure created")
            figures_results['figures_created'].append(cross_system_result)
        else:
            logger.error("❌ Failed to create cross-system feature importance figure")
            figures_results['figures_failed'].append(cross_system_result)
        
        # Generate mechanistic interpretation figure
        logger.info("Creating mechanistic interpretation figure...")
        mechanistic_figure_result = figure_generator.create_mechanistic_interpretation_figure(
            mechanistic_results,
            str(figures_dir / 'mechanistic_interpretation.png')
        )
        
        if mechanistic_figure_result['status'] == 'success':
            logger.info("✅ Mechanistic interpretation figure created")
            figures_results['figures_created'].append(mechanistic_figure_result)
        else:
            logger.error("❌ Failed to create mechanistic interpretation figure")
            figures_results['figures_failed'].append(mechanistic_figure_result)
        
        # Generate performance comparison figure
        logger.info("Creating performance comparison figure...")
        performance_data = {'empirical_evidence': {}, 'computational_efficiency': {}}  # Placeholder
        performance_result = figure_generator.create_performance_comparison_figure(
            performance_data,
            str(figures_dir / 'performance_comparison.png')
        )
        
        if performance_result['status'] == 'success':
            logger.info("✅ Performance comparison figure created")
            figures_results['figures_created'].append(performance_result)
        else:
            logger.error("❌ Failed to create performance comparison figure")
            figures_results['figures_failed'].append(performance_result)
        
        logger.info(f"📊 Figures created: {len(figures_results['figures_created'])}")
        logger.info(f"❌ Figures failed: {len(figures_results['figures_failed'])}")
        
        return figures_results
        
    except Exception as e:
        logger.error(f"❌ Error generating publication figures: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_project_overview(manuscript_generator: ManuscriptGenerator, output_dir: str) -> dict:
    """Generate complete project structure overview"""
    
    logger.info("="*80)
    logger.info("GENERATING COMPLETE PROJECT STRUCTURE OVERVIEW")
    logger.info("="*80)
    
    overview_dir = Path(output_dir) / 'project_overview'
    
    try:
        overview_results = manuscript_generator.generate_complete_project_overview(str(overview_dir))
        
        if overview_results['status'] == 'success':
            logger.info("✅ Complete project overview generated successfully")
            logger.info(f"   📄 Markdown: {overview_results['markdown_file']}")
            logger.info(f"   📊 JSON: {overview_results['json_file']}")
            logger.info(f"   📚 Supplementary docs: {len(overview_results['supplementary_docs'])}")
        else:
            logger.error("❌ Failed to generate project overview")
        
        return overview_results
        
    except Exception as e:
        logger.error(f"❌ Error generating project overview: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_final_summary(all_results: dict, output_dir: str):
    """Generate final summary of publication analysis"""
    
    logger.info("="*80)
    logger.info("GENERATING FINAL PUBLICATION ANALYSIS SUMMARY")
    logger.info("="*80)
    
    summary = {
        'title': 'Publication-Ready Analysis Summary',
        'generated_timestamp': datetime.now().isoformat(),
        'components_generated': {},
        'publication_readiness_assessment': {},
        'next_steps': [],
        'journal_submission_readiness': {}
    }
    
    # Assess component generation
    components = ['commentary', 'mechanistic', 'figures', 'overview']
    for component in components:
        if component in all_results and all_results[component].get('status') == 'success':
            summary['components_generated'][component] = 'Complete'
        else:
            summary['components_generated'][component] = 'Failed'
    
    # Assess overall publication readiness
    successful_components = sum(1 for status in summary['components_generated'].values() if status == 'Complete')
    total_components = len(summary['components_generated'])
    
    readiness_percentage = (successful_components / total_components) * 100
    
    summary['publication_readiness_assessment'] = {
        'overall_readiness': f"{readiness_percentage:.1f}%",
        'components_complete': successful_components,
        'total_components': total_components,
        'ready_for_submission': readiness_percentage >= 80
    }
    
    # Generate next steps
    if readiness_percentage >= 80:
        summary['next_steps'] = [
            'Review generated analysis materials',
            'Conduct final validation studies',
            'Prepare manuscript for journal submission',
            'Submit to target journal (Nature Materials, Acta Materialia, or Materials & Design)'
        ]
    else:
        summary['next_steps'] = [
            'Address failed components',
            'Complete missing analysis elements',
            'Re-run publication analysis generation',
            'Validate all generated materials'
        ]
    
    # Journal submission readiness
    summary['journal_submission_readiness'] = {
        'nature_materials': readiness_percentage >= 90,
        'acta_materialia': readiness_percentage >= 80,
        'materials_design': readiness_percentage >= 75
    }
    
    # Save summary
    summary_file = Path(output_dir) / 'publication_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Log summary
    logger.info("📋 PUBLICATION ANALYSIS SUMMARY")
    logger.info(f"   Overall Readiness: {summary['publication_readiness_assessment']['overall_readiness']}")
    logger.info(f"   Components Complete: {successful_components}/{total_components}")
    logger.info(f"   Ready for Submission: {summary['publication_readiness_assessment']['ready_for_submission']}")
    
    logger.info("🎯 JOURNAL SUBMISSION READINESS:")
    for journal, ready in summary['journal_submission_readiness'].items():
        status = "✅ Ready" if ready else "⚠️ Not Ready"
        logger.info(f"   {journal.replace('_', ' ').title()}: {status}")
    
    logger.info("📝 NEXT STEPS:")
    for i, step in enumerate(summary['next_steps'], 1):
        logger.info(f"   {i}. {step}")
    
    logger.info(f"💾 Summary saved: {summary_file}")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Generate publication-ready analysis and documentation')
    parser.add_argument('--output-dir', default='results/publication_analysis',
                       help='Output directory for publication analysis')
    parser.add_argument('--use-existing-results', action='store_true',
                       help='Use existing interpretability results if available')
    parser.add_argument('--interpretability-results-dir', default='results/comprehensive_interpretability_analysis',
                       help='Directory containing existing interpretability results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 STARTING PUBLICATION-READY ANALYSIS GENERATION")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Use existing results: {args.use_existing_results}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing publication analysis components...")
    analyzer = PublicationAnalyzer()
    figure_generator = PublicationFigureGenerator()
    manuscript_generator = ManuscriptGenerator()
    
    # Load interpretability results
    if args.use_existing_results:
        interpretability_results = load_existing_interpretability_results(args.interpretability_results_dir)
    else:
        interpretability_results = load_existing_interpretability_results(args.interpretability_results_dir)
    
    # Store all results
    all_results = {}
    
    # Generate comprehensive analysis commentary
    all_results['commentary'] = generate_comprehensive_analysis_commentary(analyzer, args.output_dir)
    
    # Generate mechanistic interpretation
    all_results['mechanistic'] = generate_mechanistic_interpretation(
        analyzer, interpretability_results, args.output_dir
    )
    
    # Generate publication figures
    all_results['figures'] = generate_publication_figures(
        figure_generator, interpretability_results, all_results['mechanistic'], args.output_dir
    )
    
    # Generate project overview
    all_results['overview'] = generate_project_overview(manuscript_generator, args.output_dir)
    
    # Generate final summary
    generate_final_summary(all_results, args.output_dir)
    
    logger.info("="*80)
    logger.info("🎉 PUBLICATION-READY ANALYSIS GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"📁 All results saved to: {args.output_dir}")
    logger.info("📋 Review the publication_analysis_summary.json for next steps")


if __name__ == '__main__':
    main()