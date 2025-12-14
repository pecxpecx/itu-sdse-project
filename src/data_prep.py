# src/data_prep.py

import os
import json
import datetime
import pandas as pd
import numpy as np
import warnings
from pprint import pprint
import joblib

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

# --- Define Helper Functions ---

def describe_numeric_col(x):
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    if (x.dtype == "float64") or (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode().iloc[0])
    return x

# --- Define Main Pipeline Logic ---

def run_data_processing(raw_data_path: str, artifacts_dir: str):
    
    print(f"--- Starting Data Processing ---")
    
    # 1. Setup Artifacts Directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    max_date_str = "2024-01-31"
    min_date_str = "2024-01-01"

    # 2. Read and Filter Data
    data = pd.read_csv(raw_data_path)

    max_date = pd.to_datetime(max_date_str).date()
    min_date = pd.to_datetime(min_date_str).date()
    
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    date_limits_path = os.path.join(artifacts_dir, "date_limits.json")
    with open(date_limits_path, "w") as f:
        json.dump(date_limits, f)

    # 3. Feature Selection
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen", 
            "domain", "country", "visited_learn_more_before_booking", "visited_faq"
        ],
        axis=1
    )
    
    # 4. Data Cleaning
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    data = data.dropna(axis=0, subset=["lead_indicator", "lead_id"])
    data = data[data.source == "signup"]
    
    # 5. Create Categorical Data Columns and Split
    vars_to_object = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]
    for col in vars_to_object:
        data[col] = data[col].astype("object")

    cont_vars = data.select_dtypes(include=['float64', 'int64'])
    cat_vars = data.select_dtypes(include=['object'])
    
    # 6. Outlier Handling (Clipping)
    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower = (x.mean()-2*x.std()), upper = (x.mean()+2*x.std()))
    )
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary_path = os.path.join(artifacts_dir, 'outlier_summary.csv')
    outlier_summary.to_csv(outlier_summary_path)

    # 7. Impute Data
    cont_vars = cont_vars.apply(impute_missing_values)
    
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    
    # 8. Data Standardization (MinMaxScaler)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    
    joblib.dump(value=scaler, filename=scaler_path)
    
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    # 9. Combine Data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)

    # 10. Data Drift Artifact & Initial Training Data Save
    data_columns = list(data.columns)
    columns_drift_path = os.path.join(artifacts_dir, 'columns_drift.json')
    with open(columns_drift_path, 'w+') as f:           
        json.dump(data_columns, f)
    
    training_data_path = os.path.join(artifacts_dir, 'training_data.csv')
    data.to_csv(training_data_path, index=False)

    # 11. Binning Object Columns (Strictly following notebook logic)
    data['bin_source'] = data['source']
    
    mapping = {
        'li' : 'socials', 
        'fb' : 'socials', 
        'organic': 'group1', 
        'signup': 'group1'
    }
    
    data['bin_source'] = data['source'].map(mapping)

    # 12. Save Gold Medallion Dataset
    gold_data_path = os.path.join(artifacts_dir, 'train_data_gold.csv')
    data.to_csv(gold_data_path, index=False)

    print(f"--- Data Processing Complete ---")


# --- Execution Entry Point ---

if __name__ == "__main__":
    
    RAW_DATA_INPUT_PATH = os.path.join("/", "repo", "data", "raw_data.csv") 
    ARTIFACTS_OUTPUT_DIR = "artifacts"
    
    # Check if raw data exists at the new path
    if not os.path.exists(RAW_DATA_INPUT_PATH):
        raise FileNotFoundError(
            f"Required raw data not found at: {RAW_DATA_INPUT_PATH}. "
            "Please ensure the file is in the 'data/' folder at the root level."
        )

    run_data_processing(RAW_DATA_INPUT_PATH, ARTIFACTS_OUTPUT_DIR)