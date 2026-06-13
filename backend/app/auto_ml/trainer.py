import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Try importing CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def train_and_evaluate(df: pd.DataFrame, target_col: str, problem_type: str, 
                       selected_models: list = None) -> dict:
    """
    Splits the data, trains multiple models, computes metrics, and returns the results.
    """
    # 1. Separate target and features
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Check for time column
    time_col = '__time_col__'
    has_time = time_col in df.columns
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    if has_time:
        X = X.drop(columns=[time_col])

    # 2. Train-Test Split
    if problem_type == "time_series" or has_time:
        # Chronological split (last 20% for testing)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        split_details = "Chronological split: 80% train (earliest), 20% test (latest)."
    else:
        # Standard random split
        if problem_type == "classification":
            # Stratify to ensure class representation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        split_details = "Random split: 80% train, 20% test (stratified if classification)."

    # Define model candidate dictionaries
    is_classification = problem_type == "classification"
    
    model_candidates = {}
    if is_classification:
        model_candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
        }
        if HAS_CATBOOST:
            model_candidates["CatBoost"] = CatBoostClassifier(random_state=42, verbose=0)
    else:
        # Regression or Time Series (we use regression models for time series forecasting via lag features)
        model_candidates = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVM": SVR(),
            "XGBoost": XGBRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1)
        }
        if HAS_CATBOOST:
            model_candidates["CatBoost"] = CatBoostRegressor(random_state=42, verbose=0)

    # Filter by user selection if provided
    if selected_models:
        model_candidates = {name: model for name, model in model_candidates.items() if name in selected_models}

    results = {}
    best_score = -float('inf')
    best_model_name = None
    
    # Let's count unique values in target for classification tasks (for multiclass ROC AUC)
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    for name, model in model_candidates.items():
        try:
            # Train the model and measure training time
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            # Predict and measure prediction latency
            start_pred = time.time()
            y_pred = model.predict(X_test)
            pred_time = (time.time() - start_pred) * 1000 / len(X_test)  # ms per prediction

            # Calculate metrics
            metrics = {
                "train_time_sec": round(train_time, 4),
                "pred_latency_ms": round(pred_time, 4)
            }
            
            if is_classification:
                metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                
                # Handling multi-class vs binary
                if n_classes == 2:
                    metrics["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
                    metrics["recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
                    metrics["f1"] = round(f1_score(y_test, y_pred, zero_division=0), 4)
                    
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
                        
                        # Calculate ROC Curve coordinates
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        # Downsample ROC points to keep payload small
                        indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
                        metrics["roc_curve"] = {
                            "fpr": fpr[indices].tolist(),
                            "tpr": tpr[indices].tolist()
                        }
                    except Exception:
                        metrics["roc_auc"] = 0.0
                else:
                    metrics["precision"] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                    metrics["recall"] = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                    metrics["f1"] = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                    
                    try:
                        y_prob = model.predict_proba(X_test)
                        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'), 4)
                    except Exception:
                        metrics["roc_auc"] = 0.0
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                metrics["confusion_matrix"] = cm.tolist()
                
                # The sorting score is F1 for classification
                score = metrics["f1"]

            else:
                # Regression
                metrics["r2"] = round(r2_score(y_test, y_pred), 4)
                metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
                metrics["mse"] = round(mean_squared_error(y_test, y_pred), 4)
                metrics["rmse"] = round(np.sqrt(metrics["mse"]), 4)
                
                # The sorting score is R2 for regression
                score = metrics["r2"]

            # Save the trained model instance
            results[name] = {
                "metrics": metrics,
                "model_object": model,
                "success": True
            }
            
            if score > best_score:
                best_score = score
                best_model_name = name

        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e)
            }

    return {
        "results": {name: res for name, res in results.items() if res["success"]},
        "errors": {name: res["error"] for name, res in results.items() if not res["success"]},
        "best_model": best_model_name,
        "split_details": split_details,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "features": X.columns.tolist()
    }
