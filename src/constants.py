from pathlib import Path

BASE_DIR = Path.cwd()
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
DATA_DIR = ARTIFACTS_DIR / 'data'
RUNS_DIR = ARTIFACTS_DIR / 'runs'
PLOTS_DIR = RUNS_DIR / 'plots'
TRAIN_PLOTS_DIR = PLOTS_DIR / 'train'

WEIGHTS_DIR = RUNS_DIR / 'weights'
FEATURES_PATH = DATA_DIR / 'slob_features.parquet'


TARGET_CLASSIFICATION = "SLOB_Label"
TARGET_REGRESSION = "SLOB_Volume_Proxy"
TARGET_DEMAND = "Units_Sold"


NUMERICAL_FEATURES = [
    "Units_Sold", "Stock_Level", "Lead_Time", "Safety_Stock", "Price",
    "Promotion", "Seasonal_Index", "Overall_CPI", "Food_CPI", "Non_Food_CPI",
    "Inflation_Rate", "Product_Age_Weeks", "Time_to_Expiry_Days",
    "Units_Sold_Lag1", "Units_Sold_Lag4", "Units_Sold_Lag8",
    "Rolling_Mean_4W", "Rolling_Mean_8W", "Rolling_Std_4W", "Rolling_Std_8W",
    "Consec_Zero_Weeks", "ADI_Expanding", "CV2_Expanding",
    "Zero_Demand_Ratio_Expanding", "Price_Change_Pct",
]

ABC_FEATURE = "ABC"
CATEGORICAL_FEATURES = ["Product_Category"]