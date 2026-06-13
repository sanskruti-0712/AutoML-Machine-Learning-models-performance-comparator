import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import json
from typing import List, Optional
from pydantic import BaseModel

from app.auto_ml.detector import detect_problem_type
from app.auto_ml.preprocessor import SmartPreprocessor
from app.auto_ml.trainer import train_and_evaluate
from app.auto_ml.tuner import tune_hyperparameters
from app.auto_ml.explain import get_feature_importance, compute_shap_values
from app.auto_ml.advisor import generate_ai_advice
from app.auto_ml.reporter import generate_pdf_report

app = FastAPI(title="AutoML Evaluation Platform API", version="1.0.0")

# CORS Setup for React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development; narrow down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Session Cache (representing the active workspace data)
SESSION_CACHE = {
    "df": None,
    "preprocessed_df": None,
    "preprocessor": None,
    "problem_details": None,
    "training_details": None,
    "feature_importances": {},
    "advice": None
}

class PreprocessRequest(BaseModel):
    target_col: str
    problem_type: str
    scale_data: bool = True
    detect_outliers: bool = True
    time_col: Optional[str] = None
    lags: int = 3

class TrainRequest(BaseModel):
    selected_models: Optional[List[str]] = None

class TuneRequest(BaseModel):
    model_name: str
    n_trials: int = 10

class AdviseRequest(BaseModel):
    api_key: Optional[str] = None
    perspective: Optional[str] = "expert"

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV or Excel file, reads it, and detects problem characteristics.
    """
    filename = file.filename
    try:
        # Read the file contents
        content = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
            
        # Store in global cache
        SESSION_CACHE["df"] = df
        # Reset downstream cache items
        SESSION_CACHE["preprocessed_df"] = None
        SESSION_CACHE["preprocessor"] = None
        SESSION_CACHE["problem_details"] = None
        SESSION_CACHE["training_details"] = None
        SESSION_CACHE["feature_importances"] = {}
        SESSION_CACHE["advice"] = None

        # Auto-detect problem properties
        detection = detect_problem_type(df)
        
        # Calculate column metadata
        columns_meta = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / len(df)) * 100, 2)
            col_type = str(df[col].dtype)
            columns_meta.append({
                "name": col,
                "type": col_type,
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_values": int(df[col].nunique())
            })

        # Calculate sample data
        sample_head = df.head(10).replace({pd.NA: None, float('nan'): None, float('inf'): None, float('-inf'): None}).to_dict(orient="records")

        SESSION_CACHE["problem_details"] = detection

        return {
            "filename": filename,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "detection": detection,
            "columns": columns_meta,
            "sample_data": sample_head
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file upload: {str(e)}")

@app.post("/api/preprocess")
def preprocess_dataset(req: PreprocessRequest):
    """
    Runs smart preprocessing based on selected options.
    """
    df = SESSION_CACHE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a file first.")

    try:
        # Initialize Preprocessor
        preprocessor = SmartPreprocessor(
            target_col=req.target_col,
            problem_type=req.problem_type,
            scale_data=req.scale_data,
            detect_outliers=req.detect_outliers,
            time_col=req.time_col,
            lags=req.lags
        )
        
        # Process data
        preprocessed_df, steps_log = preprocessor.fit_transform(df)
        
        # Save in cache
        SESSION_CACHE["preprocessed_df"] = preprocessed_df
        SESSION_CACHE["preprocessor"] = preprocessor
        
        # Update target details in problem details
        if SESSION_CACHE["problem_details"]:
            SESSION_CACHE["problem_details"]["detected_target"] = req.target_col
            SESSION_CACHE["problem_details"]["problem_type"] = req.problem_type
            if req.time_col:
                SESSION_CACHE["problem_details"]["datetime_columns"] = [req.time_col]
        else:
            SESSION_CACHE["problem_details"] = {
                "detected_target": req.target_col,
                "problem_type": req.problem_type,
                "total_rows": len(df),
                "total_cols": len(df.columns)
            }

        return {
            "status": "success",
            "preprocessed_shape": preprocessed_df.shape,
            "steps_log": steps_log,
            "features": [c for c in preprocessed_df.columns if c not in [req.target_col, '__time_col__']]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preprocess dataset: {str(e)}")

@app.post("/api/train")
def train_models(req: TrainRequest):
    """
    Trains selected models and compares their performance.
    """
    preprocessed_df = SESSION_CACHE.get("preprocessed_df")
    problem_details = SESSION_CACHE.get("problem_details")
    
    if preprocessed_df is None or problem_details is None:
        raise HTTPException(status_code=400, detail="No preprocessed dataset available. Run preprocessing first.")

    target_col = problem_details["detected_target"]
    problem_type = problem_details["problem_type"]

    try:
        # Train and evaluate models
        train_results = train_and_evaluate(
            df=preprocessed_df,
            target_col=target_col,
            problem_type=problem_type,
            selected_models=req.selected_models
        )
        
        # Store in cache
        SESSION_CACHE["training_details"] = train_results
        
        # Calculate feature importances for each model
        feature_importances = {}
        for model_name, res in train_results["results"].items():
            model_obj = res["model_object"]
            importances = get_feature_importance(model_obj, train_results["features"], model_name)
            feature_importances[model_name] = importances
            
        SESSION_CACHE["feature_importances"] = feature_importances

        # Build clean serialization dictionary (omit actual python model_objects)
        serializable_results = {}
        for name, res in train_results["results"].items():
            serializable_results[name] = {
                "metrics": res["metrics"],
                "success": True
            }

        return {
            "results": serializable_results,
            "errors": train_results["errors"],
            "best_model": train_results["best_model"],
            "split_details": train_results["split_details"],
            "X_train_shape": train_results["X_train_shape"],
            "X_test_shape": train_results["X_test_shape"],
            "feature_importances": feature_importances
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train models: {str(e)}")

@app.post("/api/tune")
def tune_model(req: TuneRequest):
    """
    Runs Optuna tuning on the specified model.
    """
    preprocessed_df = SESSION_CACHE.get("preprocessed_df")
    problem_details = SESSION_CACHE.get("problem_details")
    
    if preprocessed_df is None or problem_details is None:
        raise HTTPException(status_code=400, detail="No preprocessed dataset available.")

    target_col = problem_details["detected_target"]
    problem_type = problem_details["problem_type"]

    try:
        tuning_results = tune_hyperparameters(
            df=preprocessed_df,
            target_col=target_col,
            problem_type=problem_type,
            model_name=req.model_name,
            n_trials=req.n_trials
        )
        return tuning_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tune model: {str(e)}")

@app.get("/api/explain")
def explain_predictions(model_name: Optional[str] = None):
    """
    Generates SHAP explanations for the selected model.
    """
    preprocessed_df = SESSION_CACHE.get("preprocessed_df")
    problem_details = SESSION_CACHE.get("problem_details")
    training_details = SESSION_CACHE.get("training_details")

    if not preprocessed_df is not None or not training_details:
        raise HTTPException(status_code=400, detail="No trained models available for explanation.")

    best_model = training_details["best_model"]
    selected_model_name = model_name if model_name else best_model

    if selected_model_name not in training_details["results"]:
        raise HTTPException(status_code=400, detail=f"Model '{selected_model_name}' has not been trained.")

    model_obj = training_details["results"][selected_model_name]["model_object"]
    target_col = problem_details["detected_target"]
    feature_names = training_details["features"]

    # Prep X_train for SHAP background data
    X_train = preprocessed_df.drop(columns=[target_col])
    if '__time_col__' in X_train.columns:
        X_train = X_train.drop(columns=['__time_col__'])

    try:
        shap_details = compute_shap_values(
            model=model_obj,
            X_train=X_train,
            feature_names=feature_names,
            model_name=selected_model_name
        )
        return shap_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute SHAP values: {str(e)}")

@app.post("/api/advise")
def advise_models(req: AdviseRequest):
    """
    Generates LLM-based (or rule-based fallback) data science recommendations.
    """
    problem_details = SESSION_CACHE.get("problem_details")
    training_details = SESSION_CACHE.get("training_details")

    if not problem_details or not training_details:
        raise HTTPException(status_code=400, detail="Trained models are required before generating advice.")

    # Create dataset summary
    df = SESSION_CACHE["df"]
    dataset_summary = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "unique_target_values": int(problem_details.get("unique_target_values", 2)),
        "detected_target": problem_details.get("detected_target")
    }

    # Format training metrics summary
    metrics_summary = {}
    for name, res in training_details["results"].items():
        metrics_summary[name] = res["metrics"]

    try:
        advice = generate_ai_advice(
            dataset_summary=dataset_summary,
            metrics_summary=metrics_summary,
            best_model=training_details["best_model"],
            problem_type=problem_details["problem_type"],
            api_key=req.api_key,
            perspective=req.perspective or "expert"
        )
        
        # Save in cache
        SESSION_CACHE["advice"] = advice
        
        return {"advice": advice}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate advice: {str(e)}")

@app.get("/api/report")
def download_pdf_report():
    """
    Streams the generated PDF report.
    """
    df = SESSION_CACHE.get("df")
    problem_details = SESSION_CACHE.get("problem_details")
    training_details = SESSION_CACHE.get("training_details")
    preprocessor = SESSION_CACHE.get("preprocessor")
    advice = SESSION_CACHE.get("advice")

    if df is None or not problem_details or not training_details:
        raise HTTPException(status_code=400, detail="Models must be trained before generating a PDF report.")

    # Build dataset summary
    dataset_summary = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "detected_target": problem_details.get("detected_target"),
        "unique_target_values": int(problem_details.get("unique_target_values", 2))
    }

    # Preprocessing step logs
    preprocessing_steps = preprocessor.steps_log if preprocessor else []

    # Get formatted metrics summary
    metrics_summary = {}
    for name, res in training_details["results"].items():
        metrics_summary[name] = {
            "metrics": res["metrics"]
        }

    # Generate advice if not in cache
    if not advice:
        advice = generate_ai_advice(
            dataset_summary=dataset_summary,
            metrics_summary={name: res["metrics"] for name, res in training_details["results"].items()},
            best_model=training_details["best_model"],
            problem_type=problem_details["problem_type"]
        )
        SESSION_CACHE["advice"] = advice

    try:
        pdf_buffer = generate_pdf_report(
            dataset_summary=dataset_summary,
            metrics_summary=metrics_summary,
            best_model=training_details["best_model"],
            problem_type=problem_details["problem_type"],
            preprocessing_steps=preprocessing_steps,
            advisor_text=advice
        )
        
        # Stream response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=AutoML_Evaluation_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build PDF report: {str(e)}")
