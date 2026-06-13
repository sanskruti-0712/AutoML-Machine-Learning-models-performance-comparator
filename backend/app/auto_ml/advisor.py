import google.generativeai as genai
import json

def get_layman_advice(dataset_summary: dict, metrics_summary: dict, best_model: str, problem_type: str) -> str:
    """
    Generates a simplified, jargon-free explanation of results for non-technical users.
    Uses real-world analogies to explain how the models work.
    """
    is_classification = problem_type == "classification"
    rows = dataset_summary.get("total_rows", 0)
    cols = dataset_summary.get("total_cols", 0)
    target_name = dataset_summary.get("detected_target", "target")
    
    explanation_parts = []
    
    # 1. Dataset Overview
    explanation_parts.append(
        f"### 1. What is in your Dataset?\n\n"
        f"We looked at your spreadsheet, which has **{rows:,}** rows of entries and **{cols}** different columns of clues (features). "
        f"The computer is learning how to predict **'{target_name}'** based on those clues.\n"
    )
    
    if rows < 500:
        explanation_parts.append(
            "- **Small Amount of Data**: Because we have a small amount of data, the computer might easily over-memorize the examples "
            "rather than learning general rules. It's like studying for an exam by memorizing 10 questions—if the exam has different questions, you might fail. "
            "Gathering more rows would help a lot."
        )
    else:
        explanation_parts.append(
            "- **Healthy Amount of Data**: We have plenty of examples for the computer to study and learn from, making its predictions more reliable."
        )
        
    explanation_parts.append("\n---\n")

    # 2. Who won and how?
    explanation_parts.append(
        f"### 2. Which Model Won and Why?\n\n"
        f"We tested several different automated math formulas (models) on your data. "
        f"The model called **{best_model}** did the best job of predicting the correct answer.\n"
    )
    
    explanation_parts.append("#### How the Winning Model Works:")
    if "XGBoost" in best_model or "LightGBM" in best_model:
        explanation_parts.append(
            "- **The Team of Experts Analogy**: Think of the winner as a large committee of decision-makers. "
            "It builds a series of small, simple rules step-by-step. Each new rule focuses specifically on correcting the mistakes made by the previous ones. "
            "By pooling all their votes, they make highly accurate predictions, even on messy real-world data.\n font-bold"
        )
    elif "Random Forest" in best_model:
        explanation_parts.append(
            "- **The Wisdom of the Crowd Analogy**: Think of this as asking 100 people a series of 'Yes/No' questions and taking the majority vote. "
            "By averaging the decisions of many independent decision trees, it cancels out individual biases and avoids overfitting.\n"
        )
    elif "SVM" in best_model:
        explanation_parts.append(
            "- **The Boundary Line Analogy**: Think of this as drawing a clear border line on a map to separate two different countries. "
            "It looks for the widest possible safety margin between groups of data points to keep them separated.\n"
        )
    else:
        # Linear models
        explanation_parts.append(
            "- **The Trend Line Analogy**: Think of this as drawing a simple straight line through a scatter plot of dots (like showing that weight increases as height increases). "
            "It works best when relationships are simple and straightforward.\n"
        )
        
    # 3. Why others did worse
    explanation_parts.append("#### Why did other models struggle?")
    
    has_linear = any(m in metrics_summary for m in ["Logistic Regression", "Linear Regression", "Ridge Regression"])
    has_tree = any(m in metrics_summary for m in ["Random Forest", "XGBoost", "LightGBM"])
    
    if has_linear and has_tree and best_model in ["XGBoost", "LightGBM", "Random Forest"]:
        explanation_parts.append(
            "- **Relationships are Complex**: Simple trend line models (like Logistic/Linear Regression) underperformed. "
            "This tells us that your dataset has complex, curved, or overlapping relationships that cannot be separated by a simple straight line.\n"
        )
        
    if "SVM" in metrics_summary and rows > 5000:
        explanation_parts.append(
            "- **Slow Boundary Calculations**: The Boundary Line model (SVM) was slow because it has to calculate distances "
            "between every single point, which gets computationally heavy on large spreadsheets.\n"
        )

    explanation_parts.append("\n---\n")

    # 4. Next steps
    explanation_parts.append(
        "### 3. Simple Tips to Make it Better\n\n"
        "1. **Get More Clues**: Try to gather more columns of information that could influence the prediction (for example, if predicting house prices, add distance to nearest school).\n"
        "2. **Tune the Settings**: Use the 'Hyperparameter Tuning' feature to adjust the internal dials of the winning model to boost its accuracy further.\n"
        "3. **Add More Examples**: If you have a small dataset, adding more rows of examples will help the model learn more general rules."
    )
    
    return "\n".join(explanation_parts)


def get_rule_based_advice(dataset_summary: dict, metrics_summary: dict, best_model: str, problem_type: str) -> str:
    """
    Generates a rule-based expert explanation when no LLM API key is provided.
    First provides a detailed description of the uploaded dataset.
    """
    is_classification = problem_type == "classification"
    rows = dataset_summary.get("total_rows", 0)
    cols = dataset_summary.get("total_cols", 0)
    target_dist = dataset_summary.get("unique_target_values", 0)
    target_name = dataset_summary.get("detected_target", "target")
    
    explanation_parts = []
    
    # 1. Dataset Characteristics & Overview
    explanation_parts.append(
        f"### 1. Dataset Ingestion & Overview\n\n"
        f"The uploaded dataset contains **{rows:,}** observations (rows) and **{cols}** features (columns). "
        f"The platform auto-detected the target variable as **'{target_name}'** for a **{problem_type.replace('_', ' ').capitalize()}** task.\n"
    )
    
    # Custom dataset dimensions analytics
    explanation_parts.append("#### Dataset Properties & Characteristics:")
    
    if rows < 500:
        explanation_parts.append(
            "- **Small Dataset Warning**: Having fewer than 500 observations increases the risk of overfitting, "
            "especially for complex models like Random Forest and XGBoost. Linear models with regularization or shallow trees are recommended."
        )
    elif rows > 10000:
        explanation_parts.append(
            f"- **Large Sample Volume**: With {rows:,} rows, the dataset is sufficiently large to support ensemble learning "
            "like Gradient Boosting (XGBoost/LightGBM) to capture deep feature relationships without high variance risks."
        )
    else:
        explanation_parts.append(
            "- **Standard Dataset Size**: The sample size is adequate for training robust classifiers/regressors. "
            "Cross-validation split scores will be highly reliable."
        )
        
    if cols > 30:
        explanation_parts.append(
            f"- **High Dimensionality Alert**: The dataset has {cols} features. High dimensionality can trigger the "
            "'curse of dimensionality' and multicollinearity. Regularization (L1/L2 or Ridge/Lasso) or tree-based feature selection is recommended."
        )
        
    if is_classification:
        explanation_parts.append(
            f"- **Class Target Distribution**: The target contains {target_dist} distinct classes. "
            "If these classes represent uneven distributions, standard accuracy can be misleading, making F1-Score our priority metric."
        )
    else:
        explanation_parts.append(
            f"- **Continuous Target Scaling**: The target '{target_name}' is continuous. Preprocessing has separated "
            "this variable to prevent data leakage during feature transformations."
        )
        
    explanation_parts.append("\n---\n")

    # 2. Opening summary of training
    explanation_parts.append(
        f"### 2. Model Performance Analysis\n\n"
        f"After evaluating the models, **{best_model}** emerged as the top-performing estimator.\n"
    )
    
    # 3. Explaining why the model won
    explanation_parts.append("#### Why the Winning Model Outperformed Others:\n")
    if "XGBoost" in best_model or "LightGBM" in best_model:
        explanation_parts.append(
            "- **Gradient Boosting Trees**: Boosting builds trees sequentially to minimize errors. "
            "This handles complex non-linear combinations and missing values elegantly without feature transformations.\n"
            "- **Robustness to Scaling & Outliers**: Trees don't require scaled data and are robust to outlier spikes.\n"
        )
    elif "Random Forest" in best_model:
        explanation_parts.append(
            "- **Ensemble Averaging**: Bagging averages predictions across multiple decision trees, "
            "reducing overall variance and variance on noisy observations.\n"
        )
    elif "SVM" in best_model:
        explanation_parts.append(
            "- **Kernel Boundary Selection**: SVM maps features into higher-dimensional spaces using kernel tricks "
            "to establish wide margin boundaries between classes.\n"
        )
    elif "Logistic Regression" in best_model or "Linear Regression" in best_model or "Ridge" in best_model:
        explanation_parts.append(
            "- **Linear Boundary**: Simple straight decision boundary. Least prone to overfitting, "
            "and works best when feature-to-target weights are directly proportional.\n"
        )
        
    # 4. Analyze underperforming models
    explanation_parts.append("#### Analysis of Other Models:\n")
    
    has_linear = any(m in metrics_summary for m in ["Logistic Regression", "Linear Regression", "Ridge Regression"])
    has_tree = any(m in metrics_summary for m in ["Random Forest", "XGBoost", "LightGBM"])
    
    if has_linear and has_tree:
        linear_key = "Logistic Regression" if is_classification else "Linear Regression"
        if linear_key in metrics_summary and best_model in ["XGBoost", "LightGBM", "Random Forest"]:
            linear_score = metrics_summary[linear_key].get("f1" if is_classification else "r2", 0)
            best_score = metrics_summary[best_model].get("f1" if is_classification else "r2", 0)
            diff = round(best_score - linear_score, 4)
            if diff > 0.05:
                explanation_parts.append(
                    f"- **Non-Linear Relationships**: The linear model ({linear_key}) underperformed compared to {best_model} by {diff * 100:.1f}%. "
                    "This strongly suggests that the target variable has complex, non-linear dependencies on features.\n"
                )
                
    if "SVM" in metrics_summary:
        svm_time = metrics_summary["SVM"].get("train_time_sec", 0)
        rf_time = metrics_summary.get("Random Forest", {}).get("train_time_sec", 0)
        if svm_time > rf_time * 3 and rows > 5000:
            explanation_parts.append(
                f"- **Computational Complexity**: SVM took {svm_time:.2f} seconds to train. "
                "SVM computational complexity scales quadratically or cubically with the number of samples.\n"
            )
            
    if is_classification and target_dist == 2:
        for model_name, metrics in metrics_summary.items():
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            if acc - f1 > 0.15:
                explanation_parts.append(
                    f"- **Class Imbalance Alert**: For {model_name}, there is a large gap between Accuracy ({acc*100:.1f}%) and F1-Score ({f1*100:.1f}%). "
                    "This indicates class imbalance where models guess the majority class.\n"
                )
                break
                
    explanation_parts.append("\n---\n")

    # 5. Next steps / Recommendations
    explanation_parts.append(
        "### 3. Recommendations & Next Steps\n\n"
        "1. **Feature Engineering**: Since trees performed best, focus on creating interaction terms, binning continuous variables, or using target encoding for categorical columns.\n"
        "2. **Hyperparameter Tuning**: Perform Optuna search on the winning model to tune parameters like max_depth, learning_rate, and regularization weights.\n"
        "3. **Address Class Imbalance** (if applicable): Use class weight adjustments, SMOTE, or adjust the prediction threshold to improve F1-score."
    )
    
    return "\n".join(explanation_parts)

def generate_ai_advice(dataset_summary: dict, metrics_summary: dict, best_model: str, 
                       problem_type: str, api_key: str = None, perspective: str = "expert") -> str:
    """
    Main advisor router. Generates advice based on the selected perspective (expert vs layman).
    Calls Gemini API if api_key is provided, else falls back to heuristics.
    """
    is_layman = perspective == "layman"
    
    if not api_key:
        if is_layman:
            return get_layman_advice(dataset_summary, metrics_summary, best_model, problem_type)
        else:
            return get_rule_based_advice(dataset_summary, metrics_summary, best_model, problem_type)
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if is_layman:
            prompt = f"""
You are a friendly, expert Business Analyst and Educator. You are explaining the results of an AutoML run to a business manager or layman who does not know math or computer science.
Review the following dataset and training metrics:

Dataset Summary:
{json.dumps(dataset_summary, indent=2)}

Trained Model Performance Summary (Problem Type: {problem_type}):
{json.dumps(metrics_summary, indent=2)}

Winning Model: {best_model}

Please write a friendly, jargon-free overview (in markdown format). You MUST structure it as follows:

1. ### 1. What is in your Dataset?
Explain the dataset columns, size, rows, and what we are trying to predict (target: {dataset_summary.get('detected_target')}). Discuss potential advantages or limitations of this dataset in simple, clear business terms. Avoid any mathematical formulas.

2. ### 2. Which Model Won and Why?
Explain the winning model ({best_model}) and how it works using a simple real-world analogy (e.g. compare Random Forests to a committee voting, SVM to drawing a boundary safety path, etc.). Describe why it outperformed the others. Explain why other models struggled using simple non-technical analogies.

3. ### 3. Simple Tips to Make it Better
Provide 3 friendly, actionable steps the business user can take to improve the predictions (e.g. gather more columns/clues, collect more rows, run hyperparameter tuning dials).

Avoid complex words like 'collinearity', 'R-squared', 'overfitting', or 'boosting' without explaining them simply first. Make the markdown beautiful and structured.
"""
        else:
            prompt = f"""
You are an expert Data Scientist and ML Engineer. You are helping a user analyze the results of an AutoML run.
Review the following dataset and training metrics:

Dataset Summary:
{json.dumps(dataset_summary, indent=2)}

Trained Model Performance Summary (Problem Type: {problem_type}):
{json.dumps(metrics_summary, indent=2)}

Winning Model: {best_model}

Please write a comprehensive, professional explanation (in markdown format). You MUST structure it as follows:

1. ### 1. Dataset Ingestion & Overview
Provide a detailed description of the uploaded dataset:
- Analyze its dimensions (rows and columns) and potential advantages or challenges of this dataset size (e.g. small sample count prone to overfitting vs large sample size; high feature counts relative to samples).
- Mention the target variable and task type, explaining any implications of its structure (cardinality, class imbalance, or scale).

2. ### 2. Model Performance Analysis
Compare the trained models:
- Explain why the winning model ({best_model}) outperformed the others based on typical model properties (e.g. tree structures handling non-linearity vs straight boundaries, etc.).
- Analyze the limitations and failures of the underperforming models (e.g., why linear models failed, why SVM scaling was slow, etc.).

3. ### 3. Recommendations & Next Steps
Provide 3 concrete, actionable recommendations to improve performance (e.g., specific feature engineering techniques, class weight adjustments, hyperparameter search ranges).

Keep the explanation highly analytical, professional, and clear. Use markdown subheadings, bold text, and bullet lists. Do not include vague placeholders.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        err_msg = f"*AI Advisor (Gemini connection failed: {str(e)}). Displaying rule-based advice instead.*\n\n"
        if is_layman:
            return err_msg + get_layman_advice(dataset_summary, metrics_summary, best_model, problem_type)
        else:
            return err_msg + get_rule_based_advice(dataset_summary, metrics_summary, best_model, problem_type)
