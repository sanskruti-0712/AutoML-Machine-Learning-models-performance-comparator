# AutoML Evaluation Platform & AI-Powered ML Advisor ⭐⭐⭐⭐⭐

An advanced, end-to-end AutoML platform and AI-Powered Data Science Advisor. This application automates dataset ingestion, conducts problem detection, executes a smart data preprocessing pipeline, trains and evaluates multiple machine learning algorithms in parallel, visualizes SHAP explanations, tunes hyperparameters via Optuna, generates downloadable PDF evaluation reports, and uses Gemini GenAI to explain model performance.

---

## 🚀 Key Features

### 1. Automatic Problem & Time-Series Detection
- Automatically parses uploaded files (`.csv`, `.xlsx`, `.xls`) to extract dimensions, schema structures, and missing value rates.
- Evaluates the target variable cardinality to distinguish **Binary Classification**, **Multi-class Classification**, and **Continuous Regression** problems.
- Scans for date/time columns to recommend **Time-Series Split** validation and automatically engineering lag-features.

### 2. Smart Preprocessing Pipeline
- **Missing Values**: Automatically imputes numerical features using median values and categorical features with constant placeholder flags.
- **Categorical Encoding**: Dynamically applies One-Hot Encoding for low cardinality categories (<= 10) and Label Encoding for high cardinality categories.
- **Outlier Filtering**: Incorporates Isolation Forest multivariate anomaly detection to filter noisy observations.
- **Scaling**: Standardizes numerical fields (mean=0, variance=1) dynamically for distance-based estimators.
- **Time-Series Lags**: Sorts data chronologically and builds target lag variables and shifts.

### 3. Estimator Evaluation Suite
- **Classification**: Logistic Regression, Random Forest, Support Vector Classifier (SVC), XGBoost, LightGBM, and CatBoost (optional).
  - Metrics computed: Accuracy, Weighted Precision, Recall, F1-Score, ROC-AUC, ROC Curve Coordinates, and Confusion Matrix.
- **Regression**: Linear Regression, Ridge, Random Forest Regressor, SVR, XGBoost, and LightGBM.
  - Metrics computed: R² Score, Mean Absolute Error (MAE), MSE, and Root Mean Squared Error (RMSE).
- Measures and ranks estimators by **Training time (s)** and **Prediction Latency (ms)**.

### 4. Hyperparameter Tuning with Optuna
- Runs an Optuna optimization study in the background to tune the top-performing model's hyperparameter space.
- Returns "Before vs After" metric validations, optimal hyperparameter maps, and a trial history log (rendered as a convergence chart).

### 5. Explainable AI (SHAP & Feature Importance)
- Computes global feature impact (mean absolute SHAP values) and local prediction waterfall charts (showing how features push predictions away from the baseline).
- Integrates traditional Gini/Coefficient feature importances.

### 6. AI-Powered Data Science Advisor
- Connects to Google Gemini API to analyze the AutoML training metrics and explain model behavior in natural language.
- Fallback: Uses a rule-based expert heuristic engine that analyzes multicollinearity, class imbalance, and data scale when no API key is set.

### 7. Instant PDF Report Generation
- Exports a professionally styled PDF document containing dataset summary tables, preprocessing steps, model performance metrics, best model callout, and advisor insights.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.13+, FastAPI, Uvicorn, Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, SHAP, Optuna, ReportLab, Google GenerativeAI.
- **Frontend**: React 18, Vite, Tailwind CSS v4, Recharts, Lucide Icons.
- **Orchestration**: Docker, Docker Compose.
- **CI/CD**: GitHub Actions.

---

## 📂 Project Structure

```
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI workflow
├── backend/
│   ├── app/
│   │   ├── auto_ml/         # AutoML Core Logic
│   │   │   ├── advisor.py     # Rule-based & Gemini Advisor
│   │   │   ├── detector.py    # Problem type auto-detect
│   │   │   ├── explain.py     # SHAP & Feature Importance
│   │   │   ├── preprocessor.py# Smart Data Imputer/Encoder
│   │   │   ├── reporter.py    # PDF report builder
│   │   │   └── trainer.py     # Training and evaluation loop
│   │   │   └── tuner.py       # Optuna hyperparameter tuner
│   │   └── main.py          # FastAPI server entrypoint
│   ├── tests/
│   │   ├── verify.py        # Sanity check validation run
│   │   └── test_auto_ml.py  # Pytest automation tests
│   ├── Dockerfile
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main dashboard view
│   │   ├── index.css        # Tailwind v4 setup & Minimalist CSS
│   │   └── main.jsx
│   ├── Dockerfile
│   ├── vite.config.js       # Vite configuration
│   └── package.json
├── docker-compose.yml       # Docker container coordinator
└── README.md
```

---

## ⚙️ Setup & Execution

### Option A: Standard Local Run

#### 1. Start the FastAPI Backend
```bash
cd backend
python -m venv .venv
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs will be available at `http://localhost:8000/docs`.

#### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

### Option B: Docker Compose

Spin up the entire platform in containerized environments:
```bash
docker-compose up --build
```
Access the application at `http://localhost:3000` (FastAPI backend will run on port `8000`).

---

## 🧪 Testing Walkthrough

We verify the AutoML execution pipeline using automated unit tests and manual sanity scripts.

### 1. Pytest Automation Suite
We verify target parsing, missing value handling, outlier cleaning, model training, metrics calculation, and PDF exports:
```bash
cd backend
.venv\Scripts\python.exe -m pytest
```
Output:
```
======================== 4 passed, 1 warning in 10.65s ========================
```

### 2. Sanity Validation Logs
We executed a custom test script (`verify.py`) to run the entire pipeline end-to-end:
```bash
cd backend
.venv\Scripts\python.exe tests/verify.py
```
Output:
```
Starting AutoML sanity check...
[OK] All major packages imported successfully!
[OK] All local app modules imported successfully!
Running mock training run...
Detected problem: classification (binary)
Preprocessing completed. Shape: (97, 5)
Steps logged:
 - Outlier Detection: Detected and filtered out 3 outliers using Isolation Forest.
 - One-Hot Encoding: Encoded low-cardinality features using One-Hot Encoding: cat.
 - Feature Scaling: Standardized all numerical features (mean=0, variance=1) using StandardScaler.
 - Target Encoding: Encoded classification target 'target' categories: ['No', 'Yes'].
Training completed. Best model: Random Forest
 - Logistic Regression: F1=0.4211, time=0.0209s
 - Random Forest: F1=0.4706, time=0.3558s
PDF generated successfully. Size: 4489 bytes
[SUCCESS] Sanity check passed successfully!
```
"# AutoML-Machine-Learning-models-performance-comparator" 
