# scripts/05_evaluate_models.py
import sys; sys.path.append('.')
import pandas as pd
from pathlib import Path
from loguru import logger
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.interpretation.visualization import parity_plot, residual_plot

def main():
    base = Path("results/models")
    metrics_out = Path("results/metrics"); metrics_out.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures"); figs.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator()
    errx = ErrorAnalyzer()

    rows = []
    for sys_dir in base.glob("*"):
        for prop_dir in sys_dir.glob("*"):
            pred_file = prop_dir / "test_predictions.csv"
            if not pred_file.exists():
                continue
            df = pd.read_csv(pred_file)
            # Prefer ensemble column if present
            y_true = df["y_true"].values
            y_pred = (df["y_pred_ensemble"]
                      if "y_pred_ensemble" in df.columns else df.filter(like="y_pred_").mean(axis=1)).values
            m = evaluator.evaluate(y_true, y_pred, property_name=prop_dir.name)
            m.update({"ceramic_system": sys_dir.name})
            rows.append(m)

            # Plots
            parity_plot(y_true, y_pred, f"{sys_dir.name} - {prop_dir.name}",
                        figs / "predictions" / f"{sys_dir.name}_{prop_dir.name}_parity.png")
            residual_plot(y_true, y_pred, f"{sys_dir.name} - {prop_dir.name}",
                          figs / "predictions" / f"{sys_dir.name}_{prop_dir.name}_residuals.png")

            # Error tables
            err_df, stats = errx.summarize_errors(y_true, y_pred)
            errx.save_tables(
                {"errors": err_df, "stats": stats},
                prop_dir / "errors"
            )

    pd.DataFrame(rows).to_csv(metrics_out / "evaluation_summary.csv", index=False)
    logger.info("✓ Evaluation complete -> results/metrics/evaluation_summary.csv")

if __name__ == "__main__":
    main()
