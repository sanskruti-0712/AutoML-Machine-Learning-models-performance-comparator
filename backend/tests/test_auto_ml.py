import pytest
import pandas as pd
import numpy as np
import io

from app.auto_ml.detector import detect_problem_type
from app.auto_ml.preprocessor import SmartPreprocessor
from app.auto_ml.trainer import train_and_evaluate
from app.auto_ml.reporter import generate_pdf_report
from app.auto_ml.advisor import get_rule_based_advice

@pytest.fixture
def sample_classification_data():
    """Generates a mock classification dataset with some missing values and duplicates."""
    np.random.seed(42)
    n_samples = 100
    
    # 3 numerical features
    num_feat1 = np.random.normal(10, 2, n_samples)
    num_feat2 = np.random.normal(5, 5, n_samples)
    num_feat3 = np.random.normal(0, 1, n_samples)
    
    # 1 categorical feature
    cat_feat = np.random.choice(['High', 'Medium', 'Low'], size=n_samples)
    
    # Target
    target = np.random.choice(['Yes', 'No'], size=n_samples)
    
    # Introduce duplicate rows
    df = pd.DataFrame({
        'num_1': num_feat1,
        'num_2': num_feat2,
        'num_3': num_feat3,
        'cat_1': cat_feat,
        'target': target
    })
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Inject missing values
    df.loc[10:15, 'num_1'] = np.nan
    df.loc[20:25, 'cat_1'] = np.nan
    
    return df

def test_problem_type_detection(sample_classification_data):
    df = sample_classification_data
    detection = detect_problem_type(df, target_col='target')
    
    assert detection['detected_target'] == 'target'
    assert detection['problem_type'] == 'classification'
    assert detection['subtype'] == 'binary'
    assert detection['unique_target_values'] == 2
    assert detection['total_rows'] == 105

def test_smart_preprocessing(sample_classification_data):
    df = sample_classification_data
    preprocessor = SmartPreprocessor(
        target_col='target',
        problem_type='classification',
        scale_data=True,
        detect_outliers=True
    )
    
    processed_df, steps = preprocessor.fit_transform(df)
    
    # Verify duplicates were removed (105 - 5 = 100)
    # Plus outlier removal could drop a few more rows
    assert len(processed_df) <= 100
    
    # Verify target encoding
    assert 'target' in processed_df.columns
    assert processed_df['target'].dtype in [np.int32, np.int64, np.int8]
    
    # Verify no nulls remain in columns
    assert processed_df.isnull().sum().sum() == 0
    
    # Verify transforming new data works
    transformed_df = preprocessor.transform(df.iloc[:10])
    assert transformed_df.isnull().sum().sum() == 0

def test_model_training(sample_classification_data):
    df = sample_classification_data
    preprocessor = SmartPreprocessor(
        target_col='target',
        problem_type='classification'
    )
    processed_df, _ = preprocessor.fit_transform(df)
    
    # Train only standard scikit-learn models for faster test run
    selected_models = ['Logistic Regression', 'Random Forest']
    
    results = train_and_evaluate(
        df=processed_df,
        target_col='target',
        problem_type='classification',
        selected_models=selected_models
    )
    
    assert 'results' in results
    assert len(results['results']) == 2
    assert 'Logistic Regression' in results['results']
    assert 'Random Forest' in results['results']
    
    # Assert metrics are present
    lr_metrics = results['results']['Logistic Regression']['metrics']
    assert 'accuracy' in lr_metrics
    assert 'f1' in lr_metrics
    assert 'train_time_sec' in lr_metrics

def test_pdf_report_generation():
    # Mock data summaries
    dataset_summary = {
        "detected_target": "target",
        "total_rows": 100,
        "total_cols": 5,
        "unique_target_values": 2
    }
    
    metrics_summary = {
        "Logistic Regression": {
            "metrics": {
                "accuracy": 0.85,
                "f1": 0.84,
                "precision": 0.86,
                "recall": 0.82,
                "train_time_sec": 0.01
            }
        },
        "Random Forest": {
            "metrics": {
                "accuracy": 0.88,
                "f1": 0.87,
                "precision": 0.89,
                "recall": 0.85,
                "train_time_sec": 0.15
            }
        }
    }
    
    preprocessing_steps = [
        {"step": "Imputation", "details": "Imputed missing values."},
        {"step": "Scaling", "details": "Scaled features."}
    ]
    
    advice_text = get_rule_based_advice(dataset_summary, {k: v["metrics"] for k, v in metrics_summary.items()}, "Random Forest", "classification")
    
    pdf_buffer = generate_pdf_report(
        dataset_summary=dataset_summary,
        metrics_summary=metrics_summary,
        best_model="Random Forest",
        problem_type="classification",
        preprocessing_steps=preprocessing_steps,
        advisor_text=advice_text
    )
    
    # Assert buffer has content
    assert pdf_buffer.getvalue() is not None
    assert len(pdf_buffer.getvalue()) > 0
