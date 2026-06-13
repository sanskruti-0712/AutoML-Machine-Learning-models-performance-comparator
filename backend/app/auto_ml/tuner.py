import time
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Try importing CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Suppress Optuna logs to keep CLI clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(df: pd.DataFrame, target_col: str, problem_type: str, 
                         model_name: str, n_trials: int = 10) -> dict:
    """
    Runs an Optuna study to optimize hyperparameters for the selected model.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    if '__time_col__' in X.columns:
        X = X.drop(columns=['__time_col__'])

    is_classification = problem_type == "classification"
    cv_metric = "f1_weighted" if is_classification else "r2"

    # Define base score before tuning
    if is_classification:
        if model_name == "Logistic Regression":
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "Random Forest":
            base_model = RandomForestClassifier(random_state=42)
        elif model_name == "SVM":
            base_model = SVC(probability=True, random_state=42)
        elif model_name == "XGBoost":
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
        elif model_name == "LightGBM":
            base_model = LGBMClassifier(random_state=42, verbose=-1)
        elif model_name == "CatBoost" and HAS_CATBOOST:
            base_model = CatBoostClassifier(random_state=42, verbose=0)
        else:
            base_model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        if model_name == "Linear Regression":
            base_model = Ridge()  # Use Ridge as simple baseline
        elif model_name == "Ridge Regression":
            base_model = Ridge()
        elif model_name == "Random Forest":
            base_model = RandomForestRegressor(random_state=42)
        elif model_name == "SVM":
            base_model = SVR()
        elif model_name == "XGBoost":
            base_model = XGBRegressor(random_state=42)
        elif model_name == "LightGBM":
            base_model = LGBMRegressor(random_state=42, verbose=-1)
        elif model_name == "CatBoost" and HAS_CATBOOST:
            base_model = CatBoostRegressor(random_state=42, verbose=0)
        else:
            base_model = Ridge()

    # Calculate baseline score using 3-fold cross validation
    try:
        base_scores = cross_val_score(base_model, X, y, cv=3, scoring=cv_metric)
        before_score = round(float(np.mean(base_scores)), 4)
    except Exception:
        before_score = 0.0

    trial_history = []

    def objective(trial):
        params = {}
        if is_classification:
            if model_name == "Logistic Regression":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                    "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                    "max_iter": 1000,
                    "random_state": 42
                }
                # liblinear supports l1/l2; lbfgs only l2 (in standard sklearn settings)
                model = LogisticRegression(**params)
            elif model_name == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "random_state": 42
                }
                model = RandomForestClassifier(**params)
            elif model_name == "SVM":
                params = {
                    "C": trial.suggest_float("C", 0.1, 10.0),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                    "probability": True,
                    "random_state": 42
                }
                model = SVC(**params)
            elif model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "eval_metric": 'logloss',
                    "random_state": 42
                }
                model = XGBClassifier(**params)
            elif model_name == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "verbose": -1,
                    "random_state": 42
                }
                model = LGBMClassifier(**params)
            elif model_name == "CatBoost" and HAS_CATBOOST:
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 200),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "verbose": 0,
                    "random_state": 42
                }
                model = CatBoostClassifier(**params)
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            # Regression
            if model_name == "Linear Regression" or model_name == "Ridge Regression":
                params = {
                    "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True)
                }
                model = Ridge(**params)
            elif model_name == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "random_state": 42
                }
                model = RandomForestRegressor(**params)
            elif model_name == "SVM":
                params = {
                    "C": trial.suggest_float("C", 0.1, 10.0),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"])
                }
                model = SVR(**params)
            elif model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "random_state": 42
                }
                model = XGBRegressor(**params)
            elif model_name == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "verbose": -1,
                    "random_state": 42
                }
                model = LGBMRegressor(**params)
            elif model_name == "CatBoost" and HAS_CATBOOST:
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 200),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "verbose": 0,
                    "random_state": 42
                }
                model = CatBoostRegressor(**params)
            else:
                model = Ridge()

        try:
            scores = cross_val_score(model, X, y, cv=3, scoring=cv_metric)
            score = float(np.mean(scores))
        except Exception:
            score = -999.0  # Penality score if fitting fails

        # Log trial data
        trial_history.append({
            "trial_number": trial.number,
            "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()},
            "score": round(score, 4)
        })
        return score

    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    after_score = round(study.best_value, 4)

    return {
        "model_name": model_name,
        "metric_name": cv_metric,
        "before_score": before_score,
        "after_score": after_score,
        "improvement": round(after_score - before_score, 4),
        "best_params": study.best_params,
        "trial_history": trial_history
    }
