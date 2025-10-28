# src/training/hyperparameter_tuner.py
from typing import Dict, Callable
import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from loguru import logger

class HyperparameterTuner:
    """Optuna tuner for tree-based regressors with K-fold CV on R²."""

    def __init__(self, search_space: Dict, n_trials: int = 50, random_state: int = 42):
        self.search_space = search_space
        self.n_trials = n_trials
        self.random_state = random_state

    def _suggest(self, trial: optuna.trial.Trial, sp: Dict):
        params = {}
        for k, v in sp.items():
            t = v["type"]
            if t == "int":
                params[k] = trial.suggest_int(k, v["low"], v["high"], step=v.get("step"))
            elif t == "uniform":
                params[k] = trial.suggest_float(k, v["low"], v["high"])
            elif t == "loguniform":
                params[k] = trial.suggest_float(k, v["low"], v["high"], log=True)
            elif t == "categorical":
                params[k] = trial.suggest_categorical(k, v["choices"])
        return params

    def optimize(self, model_constructor: Callable, X, y, cv_splits: int = 5):
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        def objective(trial):
            params = self._suggest(trial, self.search_space)
            model = model_constructor(params)
            scores = []
            for tr, te in kf.split(X):
                Xtr, Xte = X[tr], X[te]
                ytr, yte = y[tr], y[te]
                model.train(Xtr, ytr)
                pred = model.predict(Xte)
                scores.append(r2_score(yte, pred))
            mean_r2 = float(np.mean(scores))
            return -mean_r2  # minimize negative R²

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = -study.best_value
        logger.info(f"Optuna best R²={best_score:.4f} with params={best_params}")
        return best_params, float(best_score)
