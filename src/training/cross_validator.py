# src/training/cross_validator.py
from typing import Dict
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score
from loguru import logger

class CrossValidator:
    """K-fold and Leave-One-Ceramic-Out cross-validation."""

    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def kfold(self, model, X, y) -> Dict:
        scores = []
        for tr, te in self.kf.split(X):
            model_local = model
            if hasattr(model_local, "build_model"):
                # rebuild to reset state if needed
                model_local.build_model()
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            model_local.train(Xtr, ytr)
            pred = model_local.predict(Xte)
            scores.append(r2_score(yte, pred))
        logger.info(f"KFold R²: mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
        return {"scores": scores, "mean_r2": float(np.mean(scores)), "std_r2": float(np.std(scores))}

    def leave_one_ceramic_out(self, model_factory, datasets_by_system: Dict[str, Dict[str, np.ndarray]]):
        """
        model_factory: callable returning a NEW untrained model instance each fold
        datasets_by_system: dict {system: {'X':..., 'y':...}}
        """
        systems = list(datasets_by_system.keys())
        results = {}
        for test_sys in systems:
            train_sys = [s for s in systems if s != test_sys]
            Xtr = np.vstack([datasets_by_system[s]['X'] for s in train_sys])
            ytr = np.concatenate([datasets_by_system[s]['y'] for s in train_sys])
            Xte = datasets_by_system[test_sys]['X']
            yte = datasets_by_system[test_sys]['y']
            model = model_factory()
            model.train(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = r2_score(yte, pred)
            results[test_sys] = float(r2)
            logger.info(f"LOCO {test_sys}: R²={r2:.4f}")
        return results
