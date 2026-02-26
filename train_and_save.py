"""
Training Script — Replicates the Assignment2.ipynb pipeline
Trains the Stacking Ensemble model and saves all artifacts for deployment.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

# ===========================
# STEP 1: LOAD DATA
# ===========================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

df = pd.read_csv('data/load_data.csv')
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ===========================
# STEP 2: DATA CLEANING
# ===========================
print("\n" + "=" * 60)
print("STEP 2: Cleaning data...")
print("=" * 60)

# Handle missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled {col} with median: {median_val}")

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed {before - len(df)} duplicates")

# Handle CO2 zeros
co2_col = 'CO2(tCO2)'
co2_zero_pct = (df[co2_col] == 0).sum() / len(df) * 100
print(f"  CO2 Zero%: {co2_zero_pct:.1f}%")
if co2_zero_pct > 50:
    non_zero_median = df.loc[df[co2_col] != 0, co2_col].median()
    df[co2_col] = df[co2_col].replace(0, non_zero_median)
    print(f"  CO2 zeros replaced with non-zero median: {non_zero_median}")

# Parse datetime
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format="%d-%m-%Y %H:%M")

print(f"  After cleaning: {df.shape}")

# ===========================
# STEP 3: FEATURE ENGINEERING
# ===========================
print("\n" + "=" * 60)
print("STEP 3: Feature engineering...")
print("=" * 60)

# Basic Time Features
df['Hour'] = df['Date_Time'].dt.hour
df['Day_of_Week'] = df['Date_Time'].dt.dayofweek
df['Day_of_Month'] = df['Date_Time'].dt.day
df['Month'] = df['Date_Time'].dt.month
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)

# Time Period
def get_time_period(hour):
    if 6 <= hour < 12: return 0    # Morning
    elif 12 <= hour < 18: return 1  # Afternoon
    elif 18 <= hour < 22: return 2  # Evening
    else: return 3                   # Night

df['Time_Period'] = df['Hour'].apply(get_time_period)

# Power Features
df['Power_Factor_Diff'] = df['Lagging_Current_Power_Factor'] - df['Leading_Current_Power_Factor']
df['Usage_Rate'] = df['Usage_kWh'] / (df['NSM'] + 1)
df['Reactive_Power_Ratio'] = (df['Lagging_Current_Reactive.Power_kVarh'] /
                               (df['Leading_Current_Reactive_Power_kVarh'] + 0.001))

# Cyclical encoding
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['DOW_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
df['DOW_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Power interactions
df['Total_Reactive_Power'] = (df['Lagging_Current_Reactive.Power_kVarh'] +
                               df['Leading_Current_Reactive_Power_kVarh'])
df['Reactive_Power_Diff'] = (df['Lagging_Current_Reactive.Power_kVarh'] -
                              df['Leading_Current_Reactive_Power_kVarh'])
df['Usage_kWh_squared'] = df['Usage_kWh'] ** 2
df['Usage_kWh_log'] = np.log1p(df['Usage_kWh'])
df['Power_Factor_Product'] = (df['Lagging_Current_Power_Factor'] *
                               df['Leading_Current_Power_Factor'])
df['Avg_Power_Factor'] = (df['Lagging_Current_Power_Factor'] +
                           df['Leading_Current_Power_Factor']) / 2
df['NSM_normalized'] = df['NSM'] / 86400
df['Usage_x_LagPF'] = df['Usage_kWh'] * df['Lagging_Current_Power_Factor']
df['Usage_x_LeadPF'] = df['Usage_kWh'] * df['Leading_Current_Power_Factor']

# Handle any NaN/Inf created during feature engineering
df.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols_fe = df.select_dtypes(include=[np.number]).columns
for col in num_cols_fe:
    nan_count = df[col].isnull().sum()
    if nan_count > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  Post-FE fill: {col} ({nan_count} NaN -> filled with {med:.4f})")

print(f"  Total features: {df.shape[1]}")
remaining = df.select_dtypes(include=[np.number]).isnull().sum().sum()
print(f"  Remaining NaN: {remaining}")
assert remaining == 0, f"Still have {remaining} NaN values!"

# ===========================
# STEP 4: HANDLE OUTLIERS
# ===========================
print("\n" + "=" * 60)
print("STEP 4: Handling outliers...")
print("=" * 60)

outlier_cols = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
                'Leading_Current_Reactive_Power_kVarh', 'Usage_Rate',
                'Reactive_Power_Ratio']

for col in outlier_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=Q1, upper=Q3)
        print(f"  {col}: capped to [{Q1:.2f}, {Q3:.2f}]")

# ===========================
# STEP 5: TRAIN-TEST SPLIT
# ===========================
print("\n" + "=" * 60)
print("STEP 5: Train-test split (time-based)...")
print("=" * 60)

df = df.sort_values('Date_Time').reset_index(drop=True)

last_month = df['Date_Time'].dt.to_period('M').max()
print(f"  Last month: {last_month}")

train_mask = df['Date_Time'].dt.to_period('M') != last_month
test_mask = df['Date_Time'].dt.to_period('M') == last_month

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

drop_cols = ['Date_Time', 'Load_Type']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['Load_Type']
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['Load_Type']

# Encode target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train_encoded)

print(f"  Train: {X_train.shape[0]} rows ({X_train.shape[1]} features)")
print(f"  Train (SMOTE): {X_train_smote.shape[0]} rows")
print(f"  Test: {X_test.shape[0]} rows")
print(f"  Classes: {list(le.classes_)}")

# Save feature columns
feature_columns = list(X_train.columns)
print(f"  Feature columns ({len(feature_columns)}): {feature_columns}")

# ===========================
# STEP 6: TRAIN STACKING ENSEMBLE
# ===========================
print("\n" + "=" * 60)
print("STEP 6: Training Stacking Ensemble (this may take 3-5 min)...")
print("=" * 60)

stacking_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)),
        ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 n_jobs=-1, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=15,
                                       random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=15,
                                     random_state=42, n_jobs=-1))
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3, n_jobs=-1, passthrough=False
)

stacking_model.fit(X_train_smote, y_train_smote)
stacking_pred = stacking_model.predict(X_test)

print(f"\n  STACKING ENSEMBLE RESULTS:")
print(f"  Accuracy:  {accuracy_score(y_test_encoded, stacking_pred):.4f}")
print(f"  Precision: {precision_score(y_test_encoded, stacking_pred, average='weighted'):.4f}")
print(f"  Recall:    {recall_score(y_test_encoded, stacking_pred, average='weighted'):.4f}")
print(f"  F1-Score:  {f1_score(y_test_encoded, stacking_pred, average='weighted'):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test_encoded, stacking_pred, target_names=le.classes_))

# ===========================
# STEP 7: SAVE ARTIFACTS
# ===========================
print("\n" + "=" * 60)
print("STEP 7: Saving model artifacts...")
print("=" * 60)

os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(stacking_model, 'models/stacking_model.joblib')
print(f"  Saved: models/stacking_model.joblib ({os.path.getsize('models/stacking_model.joblib') / 1024 / 1024:.1f} MB)")

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')
print(f"  Saved: models/scaler.joblib")

# Save label encoder
joblib.dump(le, 'models/label_encoder.joblib')
print(f"  Saved: models/label_encoder.joblib")

# Save feature columns
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)
print(f"  Saved: models/feature_columns.json ({len(feature_columns)} features)")

# ===========================
# STEP 8: VERIFY SAVED MODEL
# ===========================
print("\n" + "=" * 60)
print("STEP 8: Verifying saved model...")
print("=" * 60)

loaded_model = joblib.load('models/stacking_model.joblib')
loaded_le = joblib.load('models/label_encoder.joblib')

verify_pred = loaded_model.predict(X_test)
verify_acc = accuracy_score(y_test_encoded, verify_pred)
print(f"  Loaded model accuracy: {verify_acc:.4f}")
print(f"  Predictions match: {np.array_equal(stacking_pred, verify_pred)}")

decoded = loaded_le.inverse_transform(verify_pred[:5])
print(f"  Sample predictions: {decoded}")

print("\n" + "=" * 60)
print("ALL DONE! Model artifacts saved to models/ directory.")
print("=" * 60)
