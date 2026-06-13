import pandas as pd
import numpy as np
import shap

def get_feature_importance(model, feature_names: list, model_name: str) -> list:
    """
    Extracts feature importances or coefficient weights from the trained model.
    """
    importances = []
    
    try:
        # 1. Check for tree-based feature importances
        if hasattr(model, "feature_importances_"):
            imp_vals = model.feature_importances_
            
        # 2. Check for linear coefficients
        elif hasattr(model, "coef_"):
            coef = model.coef_
            # If multi-class classification, coef_ shape is (n_classes, n_features)
            if len(coef.shape) > 1:
                imp_vals = np.mean(np.abs(coef), axis=0)
            else:
                imp_vals = np.abs(coef)
                
        # 3. Fallback for other models (e.g. SVM RBF kernel or failures)
        else:
            # Equal importance fallback if not supported
            imp_vals = np.ones(len(feature_names)) / len(feature_names)

        # Normalize and pair with names
        total_imp = np.sum(imp_vals)
        if total_imp > 0:
            imp_vals = imp_vals / total_imp
            
        for name, val in zip(feature_names, imp_vals):
            importances.append({
                "feature": name,
                "importance": round(float(val), 4)
            })
            
        # Sort in descending order
        importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
        
    except Exception as e:
        importances = [{"feature": name, "importance": round(1.0/len(feature_names), 4)} for name in feature_names]

    return importances

def compute_shap_values(model, X_train: pd.DataFrame, feature_names: list, 
                       model_name: str) -> dict:
    """
    Computes global and local SHAP values for a sample of the training set.
    """
    # Downsample for performance (SHAP can be slow)
    sample_size = min(50, len(X_train))
    background_data = X_train.sample(sample_size, random_state=42) if len(X_train) > sample_size else X_train
    
    try:
        # Select best explainer
        if "Forest" in model_name or "XGBoost" in model_name or "LightGBM" in model_name or "CatBoost" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(background_data)
        else:
            # Kernel/Linear/Permutation explainer fallback
            # We predict using a wrapper if needed, or use shap.Explainer
            explainer = shap.Explainer(model.predict, background_data)
            shap_values = explainer(background_data).values

        # Handle different SHAP output formats (e.g. classification multiclass outputs list of arrays)
        if isinstance(shap_values, list):
            # For multi-class, take mean or the first class's SHAP values
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3: # (samples, features, classes)
            shap_values = np.mean(np.abs(shap_values), axis=2)
            
        # Compute mean absolute SHAP values for global feature impact
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Format for global impact chart
        global_shap = []
        for name, val in zip(feature_names, mean_abs_shap):
            global_shap.append({
                "feature": name,
                "shap_value": round(float(val), 4)
            })
        global_shap = sorted(global_shap, key=lambda x: x["shap_value"], reverse=True)

        # For local explanation (first sample in background data)
        local_sample = background_data.iloc[0]
        local_shap_vals = shap_values[0]
        
        # Handle if local_shap_vals is multidimensional
        if len(local_shap_vals.shape) > 1:
            local_shap_vals = local_shap_vals[:, 0] # Take first class
            
        local_explanation = []
        for name, val, actual_val in zip(feature_names, local_shap_vals, local_sample):
            local_explanation.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "actual_value": round(float(actual_val), 4) if isinstance(actual_val, (int, float, np.integer, np.floating)) else str(actual_val)
            })
            
        # Sort by magnitude of contribution
        local_explanation = sorted(local_explanation, key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "success": True,
            "global_shap": global_shap,
            "local_explanation": local_explanation,
            "base_value": round(float(np.mean(model.predict(background_data))), 4)
        }

    except Exception as e:
        # Fallback using feature importance mapping to simulate SHAP if library fails or is too slow
        importances = get_feature_importance(model, feature_names, model_name)
        simulated_global = []
        for imp in importances:
            simulated_global.append({
                "feature": imp["feature"],
                "shap_value": round(imp["importance"] * 0.2, 4) # Scale factor
            })
            
        return {
            "success": False,
            "error": str(e),
            "global_shap": simulated_global,
            "local_explanation": [
                {"feature": imp["feature"], "shap_value": round(imp["importance"] * 0.1, 4), "actual_value": 0.0}
                for imp in importances
            ],
            "base_value": 0.5
        }
