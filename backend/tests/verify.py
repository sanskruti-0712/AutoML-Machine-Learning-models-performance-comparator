import sys
import os

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting AutoML sanity check...")

try:
    import pandas as pd
    import numpy as np
    import fastapi
    import uvicorn
    import sklearn
    import xgboost
    import lightgbm
    import shap
    import optuna
    import reportlab
    import google.generativeai
    print("[OK] All major packages imported successfully!")
except ImportError as e:
    print(f"[ERROR] Import failed: {str(e)}")
    sys.exit(1)

try:
    from app.auto_ml.detector import detect_problem_type
    from app.auto_ml.preprocessor import SmartPreprocessor
    from app.auto_ml.trainer import train_and_evaluate
    from app.auto_ml.advisor import get_rule_based_advice
    from app.auto_ml.reporter import generate_pdf_report
    print("[OK] All local app modules imported successfully!")
except Exception as e:
    print(f"[ERROR] Local app import failed: {str(e)}")
    sys.exit(1)

# Generate dummy data
print("Running mock training run...")
np.random.seed(42)
df = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'cat': np.random.choice(['A', 'B'], size=100),
    'target': np.random.choice(['Yes', 'No'], size=100)
})

# Detect
det = detect_problem_type(df, 'target')
print(f"Detected problem: {det['problem_type']} ({det['subtype']})")

# Preprocess
pre = SmartPreprocessor('target', 'classification')
df_proc, steps = pre.fit_transform(df)
print(f"Preprocessing completed. Shape: {df_proc.shape}")
print("Steps logged:")
for s in steps:
    print(f" - {s['step']}: {s['details']}")

# Train
results = train_and_evaluate(df_proc, 'target', 'classification', ['Logistic Regression', 'Random Forest'])
print(f"Training completed. Best model: {results['best_model']}")
for name, res in results['results'].items():
    print(f" - {name}: F1={res['metrics']['f1']:.4f}, time={res['metrics']['train_time_sec']:.4f}s")

# Report
advice = get_rule_based_advice(det, {k: v["metrics"] for k, v in results["results"].items()}, results["best_model"], "classification")
pdf = generate_pdf_report(det, results["results"], results["best_model"], "classification", steps, advice)
print(f"PDF generated successfully. Size: {len(pdf.getvalue())} bytes")

print("[SUCCESS] Sanity check passed successfully!")
