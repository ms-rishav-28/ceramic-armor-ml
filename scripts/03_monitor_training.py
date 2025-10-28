#!/usr/bin/env python3
"""
Training Monitoring Script for Ceramic Armor ML Pipeline.

Adds real-time progress monitoring to existing trainer.py, shows live R² scores
during existing cross-validation, implements time estimates for existing training
processes, and adds memory usage monitoring for existing model training.

Requirements: 3.4 - Training monitoring for existing system
"""

import sys
import time
import threading
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.config_loader import load_config
    from src.training.trainer import Trainer
    from src.training.cross_validator import CrossValidator
    from src.models.xgboost_model import XGBoostModel
    from src.models.catboost_model import CatBoostModel
    from src.models.random_forest_model import RandomForestModel
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    sys.exit(1)


class TrainingMonitor:
    """Real-time training monitoring and performance tracking."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.training_metrics = {}
        self.performance_history = []
        self.memory_history = []
        self.start_time = None
        
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
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🔍 Training monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        logger.info("⏹️  Training monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                current_time = time.time() - self.start_time
                
                metrics = {
                    'timestamp': current_time,
                    'memory_used_gb': memory_info.used / (1024**3),
                    'memory_percent': memory_info.percent,
                    'cpu_percent': cpu_percent,
                    'available_memory_gb': memory_info.available / (1024**3)
                }
                
                self.memory_history.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(10)
    
    def create_enhanced_trainer(self, trainer_class=None) -> 'MonitoredTrainer':
        """Create a trainer with monitoring capabilities."""
        if trainer_class is None:
            trainer_class = Trainer
        
        return MonitoredTrainer(trainer_class, self)
    
    def log_training_progress(self, 
                            model_name: str, 
                            fold: int, 
                            epoch: int, 
                            train_score: float, 
                            val_score: float,
                            property_name: str = "unknown") -> None:
        """Log training progress with real-time updates."""
        
        current_time = time.time() - self.start_time if self.start_time else 0
        
        progress_info = {
            'timestamp': current_time,
            'model_name': model_name,
            'property': property_name,
            'fold': fold,
            'epoch': epoch,
            'train_r2': train_score,
            'val_r2': val_score,
            'memory_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        self.performance_history.append(progress_info)
        
        # Real-time console output
        elapsed_str = self._format_time(current_time)
        logger.info(f"📊 {model_name} | {property_name} | Fold {fold} | Epoch {epoch} | "
                   f"Train R²: {train_score:.4f} | Val R²: {val_score:.4f} | "
                   f"Time: {elapsed_str}")
        
        # Update live plots every 10 iterations
        if len(self.performance_history) % 10 == 0:
            self._update_live_plots()
    
    def estimate_remaining_time(self, 
                              current_fold: int, 
                              total_folds: int,
                              current_epoch: int, 
                              total_epochs: int) -> str:
        """Estimate remaining training time."""
        
        if not self.start_time or len(self.performance_history) < 2:
            return "Calculating..."
        
        current_time = time.time() - self.start_time
        
        # Calculate progress
        fold_progress = current_fold / total_folds
        epoch_progress = current_epoch / total_epochs
        total_progress = (fold_progress + epoch_progress / total_folds)
        
        if total_progress <= 0:
            return "Calculating..."
        
        # Estimate total time
        estimated_total_time = current_time / total_progress
        remaining_time = estimated_total_time - current_time
        
        return self._format_time(remaining_time)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _update_live_plots(self) -> None:
        """Update live training plots."""
        try:
            if len(self.performance_history) < 5:
                return
            
            # Create plots directory
            plots_dir = self.project_root / 'results' / 'figures' / 'training_monitor'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot training progress
            self._plot_training_progress(plots_dir)
            
            # Plot memory usage
            self._plot_memory_usage(plots_dir)
            
        except Exception as e:
            logger.warning(f"Failed to update live plots: {e}")
    
    def _plot_training_progress(self, plots_dir: Path) -> None:
        """Plot training progress over time."""
        df = pd.DataFrame(self.performance_history)
        
        if len(df) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot by model and property
        for (model, prop), group in df.groupby(['model_name', 'property']):
            plt.plot(group['timestamp'], group['val_r2'], 
                    label=f"{model} - {prop}", marker='o', markersize=3)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Validation R²')
        plt.title('Real-time Training Progress', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add target line
        target_r2 = self.config.get('targets', {}).get('mechanical_r2', 0.85)
        plt.axhline(y=target_r2, color='red', linestyle='--', alpha=0.7, 
                   label=f'Target R² = {target_r2}')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, plots_dir: Path) -> None:
        """Plot memory usage over time."""
        if len(self.memory_history) < 5:
            return
        
        df = pd.DataFrame(self.memory_history)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['memory_used_gb'], color='blue', linewidth=2)
        plt.ylabel('Memory Used (GB)')
        plt.title('System Resource Monitoring', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['cpu_percent'], color='orange', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'resource_usage.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training monitoring report."""
        logger.info("📋 Generating Training Monitoring Report")
        
        if not self.performance_history:
            logger.warning("No training data to report")
            return {}
        
        df = pd.DataFrame(self.performance_history)
        memory_df = pd.DataFrame(self.memory_history) if self.memory_history else pd.DataFrame()
        
        # Calculate statistics
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': time.time() - self.start_time if self.start_time else 0,
            'total_iterations': len(self.performance_history),
            'models_trained': df['model_name'].nunique() if not df.empty else 0,
            'properties_trained': df['property'].nunique() if not df.empty else 0,
            'performance_summary': {},
            'resource_usage': {},
            'recommendations': []
        }
        
        if not df.empty:
            # Performance summary by model
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                report['performance_summary'][model] = {
                    'best_val_r2': float(model_data['val_r2'].max()),
                    'final_val_r2': float(model_data['val_r2'].iloc[-1]),
                    'avg_val_r2': float(model_data['val_r2'].mean()),
                    'iterations': len(model_data),
                    'convergence_rate': self._calculate_convergence_rate(model_data)
                }
        
        if not memory_df.empty:
            # Resource usage summary
            report['resource_usage'] = {
                'peak_memory_gb': float(memory_df['memory_used_gb'].max()),
                'avg_memory_gb': float(memory_df['memory_used_gb'].mean()),
                'peak_cpu_percent': float(memory_df['cpu_percent'].max()),
                'avg_cpu_percent': float(memory_df['cpu_percent'].mean())
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_training_recommendations(report)
        
        # Save report
        report_path = self.project_root / 'logs' / 'training_monitoring_report.yaml'
        try:
            with open(report_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Training report saved: {report_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
        
        return report
    
    def _calculate_convergence_rate(self, model_data: pd.DataFrame) -> float:
        """Calculate how quickly the model converges."""
        if len(model_data) < 10:
            return 0.0
        
        # Calculate improvement rate over last 50% of training
        mid_point = len(model_data) // 2
        early_r2 = model_data['val_r2'].iloc[:mid_point].mean()
        late_r2 = model_data['val_r2'].iloc[mid_point:].mean()
        
        return float(late_r2 - early_r2)
    
    def _generate_training_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training monitoring."""
        recommendations = []
        
        # Memory recommendations
        resource_usage = report.get('resource_usage', {})
        peak_memory = resource_usage.get('peak_memory_gb', 0)
        
        if peak_memory > 12:
            recommendations.append("High memory usage detected - consider reducing batch size or using data streaming")
        
        # Performance recommendations
        performance = report.get('performance_summary', {})
        
        for model, stats in performance.items():
            best_r2 = stats.get('best_val_r2', 0)
            target_r2 = self.config.get('targets', {}).get('mechanical_r2', 0.85)
            
            if best_r2 < target_r2:
                recommendations.append(f"{model} not meeting target R² ({best_r2:.3f} < {target_r2}) - consider hyperparameter tuning")
            
            convergence = stats.get('convergence_rate', 0)
            if convergence < 0.01:
                recommendations.append(f"{model} showing poor convergence - consider adjusting learning rate or regularization")
        
        # Training duration recommendations
        duration = report.get('training_duration', 0)
        if duration > 3600:  # More than 1 hour
            recommendations.append("Long training time - consider early stopping or model simplification")
        
        return recommendations


class MonitoredTrainer:
    """Wrapper for existing trainer with monitoring capabilities."""
    
    def __init__(self, trainer_class, monitor: TrainingMonitor):
        self.trainer_class = trainer_class
        self.monitor = monitor
        self.trainer = None
        
    def train(self, *args, **kwargs):
        """Train with monitoring."""
        self.monitor.start_monitoring()
        
        try:
            # Create trainer instance
            self.trainer = self.trainer_class(*args, **kwargs)
            
            # Monkey patch the trainer's methods to add monitoring
            self._add_monitoring_hooks()
            
            # Start training
            result = self.trainer.train()
            
            return result
            
        finally:
            self.monitor.stop_monitoring()
    
    def _add_monitoring_hooks(self):
        """Add monitoring hooks to trainer methods."""
        if hasattr(self.trainer, 'train_model'):
            original_train_model = self.trainer.train_model
            
            def monitored_train_model(model, X_train, y_train, X_val, y_val, property_name="unknown"):
                model_name = model.__class__.__name__
                
                # Add callback for iterative models
                if hasattr(model, 'fit'):
                    if 'XGBoost' in model_name:
                        return self._monitor_xgboost_training(
                            model, X_train, y_train, X_val, y_val, property_name
                        )
                    elif 'CatBoost' in model_name:
                        return self._monitor_catboost_training(
                            model, X_train, y_train, X_val, y_val, property_name
                        )
                    else:
                        return self._monitor_sklearn_training(
                            model, X_train, y_train, X_val, y_val, property_name
                        )
                
                return original_train_model(model, X_train, y_train, X_val, y_val, property_name)
            
            self.trainer.train_model = monitored_train_model
    
    def _monitor_xgboost_training(self, model, X_train, y_train, X_val, y_val, property_name):
        """Monitor XGBoost training with callbacks."""
        import xgboost as xgb
        from sklearn.metrics import r2_score
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Training parameters
        params = model.get_params()
        
        # Custom callback for monitoring
        def monitoring_callback(env):
            if env.iteration % 10 == 0:  # Log every 10 iterations
                train_pred = env.model.predict(dtrain)
                val_pred = env.model.predict(dval)
                
                train_r2 = r2_score(y_train, train_pred)
                val_r2 = r2_score(y_val, val_pred)
                
                self.monitor.log_training_progress(
                    model_name="XGBoost",
                    fold=0,  # Would need to be passed from cross-validation
                    epoch=env.iteration,
                    train_score=train_r2,
                    val_score=val_r2,
                    property_name=property_name
                )
        
        # Train with monitoring
        model_xgb = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=[monitoring_callback],
            verbose_eval=False
        )
        
        # Convert back to sklearn format
        model.set_params(**params)
        model._Booster = model_xgb
        
        return model
    
    def _monitor_catboost_training(self, model, X_train, y_train, X_val, y_val, property_name):
        """Monitor CatBoost training."""
        from catboost import Pool
        from sklearn.metrics import r2_score
        
        # Create CatBoost pools
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        # Custom callback class
        class MonitoringCallback:
            def __init__(self, monitor, property_name):
                self.monitor = monitor
                self.property_name = property_name
                self.iteration = 0
            
            def after_iteration(self, info):
                if self.iteration % 10 == 0:
                    train_r2 = 1 - info.metrics['learn']['RMSE'][-1] / np.var(y_train)
                    val_r2 = 1 - info.metrics['validation']['RMSE'][-1] / np.var(y_val)
                    
                    self.monitor.log_training_progress(
                        model_name="CatBoost",
                        fold=0,
                        epoch=self.iteration,
                        train_score=train_r2,
                        val_score=val_r2,
                        property_name=self.property_name
                    )
                
                self.iteration += 1
                return True
        
        # Train with monitoring
        model.fit(
            train_pool,
            eval_set=val_pool,
            callbacks=[MonitoringCallback(self.monitor, property_name)],
            verbose=False
        )
        
        return model
    
    def _monitor_sklearn_training(self, model, X_train, y_train, X_val, y_val, property_name):
        """Monitor sklearn model training (simpler monitoring)."""
        from sklearn.metrics import r2_score
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # Log final scores
        self.monitor.log_training_progress(
            model_name=model.__class__.__name__,
            fold=0,
            epoch=1,
            train_score=train_r2,
            val_score=val_r2,
            property_name=property_name
        )
        
        return model


def main():
    """Main entry point for testing training monitoring."""
    logger.info("🚀 Training Monitoring Test")
    logger.info("=" * 50)
    
    # Create monitor
    monitor = TrainingMonitor()
    
    # Test monitoring with dummy data
    logger.info("Testing monitoring capabilities...")
    
    monitor.start_monitoring()
    
    # Simulate training progress
    for epoch in range(20):
        train_r2 = 0.5 + 0.3 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 0.02)
        val_r2 = 0.4 + 0.35 * (1 - np.exp(-epoch / 8)) + np.random.normal(0, 0.03)
        
        monitor.log_training_progress(
            model_name="TestModel",
            fold=1,
            epoch=epoch,
            train_score=train_r2,
            val_score=val_r2,
            property_name="youngs_modulus"
        )
        
        time.sleep(0.5)  # Simulate training time
    
    monitor.stop_monitoring()
    
    # Generate report
    report = monitor.generate_training_report()
    
    logger.info("✅ Training monitoring test completed")
    logger.info(f"📊 Report generated with {len(monitor.performance_history)} data points")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())