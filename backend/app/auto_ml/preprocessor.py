import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class SmartPreprocessor:
    def __init__(self, target_col: str, problem_type: str, scale_data: bool = True, 
                 detect_outliers: bool = True, time_col: str = None, lags: int = 3):
        self.target_col = target_col
        self.problem_type = problem_type
        self.scale_data = scale_data
        self.detect_outliers = detect_outliers
        self.time_col = time_col
        self.lags = lags
        
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.one_hot_encoder = None
        self.label_encoders = {}
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.one_hot_cols = []
        self.label_encoded_cols = []
        self.steps_log = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Fits estimators and transforms the DataFrame, keeping a detailed log.
        Returns the preprocessed DataFrame and log steps.
        """
        self.steps_log = []
        df_clean = df.copy()
        
        # 1. Remove duplicate rows
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            self.steps_log.append({
                "step": "Duplicate Removal",
                "details": f"Detected and removed {dup_count} duplicate rows."
            })
            
        # 2. Sort by time if it's a Time Series problem
        if self.problem_type == "time_series" and self.time_col:
            if self.time_col in df_clean.columns:
                df_clean[self.time_col] = pd.to_datetime(df_clean[self.time_col])
                df_clean = df_clean.sort_values(by=self.time_col).reset_index(drop=True)
                self.steps_log.append({
                    "step": "Time Series Sorting",
                    "details": f"Sorted dataset chronologically by time column: '{self.time_col}'."
                })

        # Separate target and features
        if self.target_col in df_clean.columns:
            y = df_clean[self.target_col]
            X = df_clean.drop(columns=[self.target_col])
        else:
            y = None
            X = df_clean

        # Remove the time column from features to avoid passing raw datetime to standard models
        if self.time_col and self.time_col in X.columns:
            X_time = X[self.time_col]
            X = X.drop(columns=[self.time_col])
        else:
            X_time = None

        # Identify numerical and categorical features
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 3. Handle missing values
        # Numerical Columns
        if self.numeric_cols:
            self.num_imputer = SimpleImputer(strategy='median')
            X[self.numeric_cols] = self.num_imputer.fit_transform(X[self.numeric_cols])
            null_count = df[self.numeric_cols].isnull().sum().sum()
            if null_count > 0:
                self.steps_log.append({
                    "step": "Numerical Imputation",
                    "details": f"Imputed {null_count} missing values in numerical columns using median."
                })
        
        # Categorical Columns
        if self.categorical_cols:
            self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
            X[self.categorical_cols] = self.cat_imputer.fit_transform(X[self.categorical_cols])
            null_count = df[self.categorical_cols].isnull().sum().sum()
            if null_count > 0:
                self.steps_log.append({
                    "step": "Categorical Imputation",
                    "details": f"Imputed {null_count} missing values in categorical columns with label 'Missing'."
                })

        # 4. Outlier Detection (only on numerical columns)
        if self.detect_outliers and self.numeric_cols and len(df_clean) > 20:
            iso = IsolationForest(contamination=0.03, random_state=42)
            outliers = iso.fit_predict(X[self.numeric_cols])
            outlier_mask = outliers == -1
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                # Remove outliers
                X = X[~outlier_mask].reset_index(drop=True)
                if y is not None:
                    y = y[~outlier_mask].reset_index(drop=True)
                if X_time is not None:
                    X_time = X_time[~outlier_mask].reset_index(drop=True)
                self.steps_log.append({
                    "step": "Outlier Detection",
                    "details": f"Detected and filtered out {outlier_count} outliers using Isolation Forest (contamination=3%)."
                })

        # 5. Time Series Lag Feature Engineering
        if self.problem_type == "time_series" and self.target_col in df_clean.columns:
            # We want to create lag features of the target variable if we have target
            # To do this correctly, we add target lags to features
            lag_log_details = []
            for lag in range(1, self.lags + 1):
                lag_name = f"{self.target_col}_lag_{lag}"
                # Shift y to create lags
                X[lag_name] = y.shift(lag)
                lag_log_details.append(lag_name)
            
            # Since shifting creates NaN values at the beginning, we backfill them or drop them
            X = X.bfill()
            self.numeric_cols.extend(lag_log_details)
            self.steps_log.append({
                "step": "Time Series Lags",
                "details": f"Generated target lags: {', '.join(lag_log_details)} and backfilled initial NaNs."
            })

        # 6. Categorical Encoding
        X_encoded = pd.DataFrame(index=X.index)
        
        # One-Hot Encoding for low cardinality (<= 10 categories)
        oh_candidates = [col for col in self.categorical_cols if X[col].nunique() <= 10]
        # Label Encoding for high cardinality (> 10 categories)
        le_candidates = [col for col in self.categorical_cols if X[col].nunique() > 10]

        if oh_candidates:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            oh_data = self.one_hot_encoder.fit_transform(X[oh_candidates])
            oh_names = self.one_hot_encoder.get_feature_names_out(oh_candidates)
            oh_df = pd.DataFrame(oh_data, columns=oh_names, index=X.index)
            X_encoded = pd.concat([X_encoded, oh_df], axis=1)
            self.one_hot_cols = oh_names.tolist()
            self.steps_log.append({
                "step": "One-Hot Encoding",
                "details": f"Encoded low-cardinality features using One-Hot Encoding: {', '.join(oh_candidates)}."
            })

        for col in le_candidates:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            self.label_encoded_cols.append(col)
            
        if le_candidates:
            self.steps_log.append({
                "step": "Label Encoding",
                "details": f"Encoded high-cardinality features using Label Encoding: {', '.join(le_candidates)}."
            })

        # Keep original numerical features
        if self.numeric_cols:
            # Join with numeric columns
            X_encoded = pd.concat([X_encoded, X[self.numeric_cols]], axis=1)

        # 7. Scaling (only on numerical columns, which now includes lag variables and encoded columns)
        if self.scale_data and self.numeric_cols:
            self.scaler = StandardScaler()
            X_scaled_arr = self.scaler.fit_transform(X_encoded[self.numeric_cols])
            X_scaled_df = pd.DataFrame(X_scaled_arr, columns=self.numeric_cols, index=X_encoded.index)
            # Reassemble DataFrame with scaled columns and categorical columns
            for col in self.numeric_cols:
                X_encoded[col] = X_scaled_df[col]
            
            self.steps_log.append({
                "step": "Feature Scaling",
                "details": "Standardized all numerical features (mean=0, variance=1) using StandardScaler."
            })

        # Re-attach target
        if y is not None:
            # Handle classification target encoding if it's string/object
            if self.problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
                target_le = LabelEncoder()
                y_encoded = target_le.fit_transform(y)
                self.label_encoders[self.target_col] = target_le
                y = pd.Series(y_encoded, name=self.target_col, index=X_encoded.index)
                self.steps_log.append({
                    "step": "Target Encoding",
                    "details": f"Encoded classification target '{self.target_col}' categories: {list(target_le.classes_)}."
                })
            X_encoded[self.target_col] = y

        # Re-attach time column if present
        if X_time is not None:
            X_encoded['__time_col__'] = X_time

        return X_encoded, self.steps_log

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or new data using fitted objects.
        """
        df_clean = df.copy()
        
        # Sort by time if time series
        if self.problem_type == "time_series" and self.time_col:
            if self.time_col in df_clean.columns:
                df_clean[self.time_col] = pd.to_datetime(df_clean[self.time_col])
                df_clean = df_clean.sort_values(by=self.time_col).reset_index(drop=True)

        if self.target_col in df_clean.columns:
            y = df_clean[self.target_col]
            X = df_clean.drop(columns=[self.target_col])
        else:
            y = None
            X = df_clean

        if self.time_col and self.time_col in X.columns:
            X_time = X[self.time_col]
            X = X.drop(columns=[self.time_col])
        else:
            X_time = None

        # Apply numerical imputer
        if self.numeric_cols and self.num_imputer:
            # Only impute columns that were numeric originally
            orig_numeric = [col for col in self.numeric_cols if col in X.columns]
            if orig_numeric:
                X[orig_numeric] = self.num_imputer.transform(X[orig_numeric])
        
        # Apply categorical imputer
        if self.categorical_cols and self.cat_imputer:
            orig_categorical = [col for col in self.categorical_cols if col in X.columns]
            if orig_categorical:
                X[orig_categorical] = self.cat_imputer.transform(X[orig_categorical])

        # Generate Time Series Lags if needed
        if self.problem_type == "time_series" and y is not None:
            for lag in range(1, self.lags + 1):
                lag_name = f"{self.target_col}_lag_{lag}"
                X[lag_name] = y.shift(lag)
            X = X.bfill()

        # Categorical Encoding
        X_encoded = pd.DataFrame(index=X.index)

        # One-Hot Encoding
        oh_candidates = [col for col in self.categorical_cols if col in X.columns and col not in self.label_encoders]
        if oh_candidates and self.one_hot_encoder:
            oh_data = self.one_hot_encoder.transform(X[oh_candidates])
            oh_df = pd.DataFrame(oh_data, columns=self.one_hot_cols, index=X.index)
            X_encoded = pd.concat([X_encoded, oh_df], axis=1)

        # Label Encoding
        for col in self.label_encoders:
            if col != self.target_col and col in X.columns:
                le = self.label_encoders[col]
                # Handle unknown classes by mapping to -1
                X_encoded[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        # Add numerical columns
        if self.numeric_cols:
            available_num_cols = [c for c in self.numeric_cols if c in X.columns]
            X_encoded = pd.concat([X_encoded, X[available_num_cols]], axis=1)

        # Apply scaling
        if self.scale_data and self.scaler and self.numeric_cols:
            # Ensure order of columns matches scaler fit
            X_scaled_arr = self.scaler.transform(X_encoded[self.numeric_cols])
            X_scaled_df = pd.DataFrame(X_scaled_arr, columns=self.numeric_cols, index=X_encoded.index)
            for col in self.numeric_cols:
                X_encoded[col] = X_scaled_df[col]

        # Re-attach target
        if y is not None:
            if self.problem_type == "classification" and self.target_col in self.label_encoders:
                target_le = self.label_encoders[self.target_col]
                y_encoded = y.map(lambda s: target_le.transform([s])[0] if s in target_le.classes_ else -1)
                y = pd.Series(y_encoded, name=self.target_col, index=X_encoded.index)
            X_encoded[self.target_col] = y

        if X_time is not None:
            X_encoded['__time_col__'] = X_time

        return X_encoded
