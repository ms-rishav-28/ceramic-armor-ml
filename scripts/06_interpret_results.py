# scripts/06_interpret_results.py
import sys; sys.path.append('.')
import joblib
import pandas as pd
from pathlib import Path
from loguru import logger
from src.interpretation.shap_analyzer import SHAPAnalyzer
from src.interpretation.materials_insights import interpret_feature_ranking

def main():
    model_root = Path("results/models")
    fig_root = Path("results/figures/shap"); fig_root.mkdir(parents=True, exist_ok=True)
    insights = []

    for sys_dir in model_root.glob("*"):
        for prop_dir in sys_dir.glob("*"):
            # Load model bundle
            model_files = list(prop_dir.glob("*_model.pkl"))
            if not model_files:
                continue
            bundle = joblib.load(model_files[0])  # any model; we’ll explain ensemble via best base if needed
            model = bundle["model"]
            target = bundle.get("target_name", prop_dir.name)

            # Load test predictions to reconstruct X_test shape (stored alongside in predictions)
            pred_file = prop_dir / "test_predictions.csv"
            if not pred_file.exists():
                continue
            dfp = pd.read_csv(pred_file)
            # SHAP needs the features used for training; store feature_names in bundle
            feat_names = bundle.get("feature_names", [f"f{i}" for i in range(len(dfp.columns)-1)])

            # This simple example uses y_true only to size SHAP background; replace with cached X_test if persisted
            X_dummy = pd.DataFrame(columns=feat_names)  # empty scaffold
            # Fallback: skip if we didn’t persist feature matrix
            if X_dummy.shape[1] == 0:
                logger.warning(f"No feature matrix persisted for SHAP at {prop_dir}; skipping")
                continue

            analyzer = SHAPAnalyzer(model, model_type="tree")
            # Cannot compute without X; in a full run, persist X_test_scaled to prop_dir / 'X_test.npy'
            x_npy = prop_dir / "X_test.npy"
            if not x_npy.exists():
                logger.warning(f"Missing {x_npy} for SHAP; skipping")
                continue
            import numpy as np
            X = np.load(x_npy)
            analyzer.create_explainer(X_background=X[:min(1000, len(X))], feature_names=feat_names)
            analyzer.generate_all_plots(X, output_dir=fig_root / f"{sys_dir.name}_{prop_dir.name}", top_features=10)

            # Feature importance table (if model exposes it)
            try:
                imp = model.get_score(importance_type='gain')
                imp_df = pd.DataFrame({'feature': list(imp.keys()), 'importance': list(imp.values())}).sort_values('importance', ascending=False)
            except Exception:
                try:
                    vals = model.feature_importances_
                    imp_df = pd.DataFrame({'feature': feat_names, 'importance': vals}).sort_values('importance', ascending=False)
                except Exception:
                    imp_df = pd.DataFrame(columns=["feature","importance"])

            text = interpret_feature_ranking(imp_df, top_k=10)
            insights.append({
                "ceramic_system": sys_dir.name,
                "property": prop_dir.name,
                "insights": text
            })

    out_csv = Path("results/reports"); out_csv.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(insights).to_csv(out_csv / "materials_insights.csv", index=False)
    logger.info("✓ SHAP interpretation complete -> results/reports/materials_insights.csv")

if __name__ == "__main__":
    main()
