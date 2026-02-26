# LIVE SITE
🔗 **[https://load-type-predictor.onrender.com](https://load-type-predictor.onrender.com)**

<br>

# Load Type Predictor

A machine learning model that predicts power system **Load Type** (Light / Medium / Maximum) from energy consumption data. Deployed as a FastAPI web service on Render.

## Problem Statement
Predict the `Load_Type` of a steel industry power system using historical data. Classification into 3 classes:
- **Light_Load**
- **Medium_Load**
- **Maximum_Load**

## Dataset
- **35,041 rows** × 9 columns (Jan–Dec 2018)
- Features: `Usage_kWh`, `Lagging/Leading Reactive Power`, `CO2`, `Power Factors`, `NSM`
- Validation: **Last month (Dec 2018) as test set** (time-based split)

## Model Pipeline

```
┌────────────┐   ┌────────────────┐   ┌───────────────────┐   ┌────────────────┐   ┌────────────────┐    ┌─────────┐   ┌─────────────────┐   ┌──────────────────┐
│  Raw Data  │─▶│ Preprocessing  │──▶│ Feature Engineer  │─▶│ Outlier Handle │─▶│ Train/Test Split│─▶ │  SMOTE  │──▶│ 7 Models Trained│──▶│🏆Stacking Ensem.│
│  (35,041)  │   │• Median impute │   │• 31 features      │   │• 1st-99th %ile │   │• Time-based     │   │ Balance │   │ LR | RF | XGB   │   │ Accuracy: 93.95% │
│            │   │• Remove dupes  │   │• Time + Cyclical  │   │  capping       │   │• Last month     │   │ classes │   │ LGBM | ET       │   │ F1-Score: 93.99% │
│            │   │• CO2 zero fix  │   │• Power interact.  │   │                │   │  = test set     │   │         │   │ Voting|Stacking │   │                  │
└────────────┘   └────────────────┘   └───────────────────┘   └────────────────┘   └────────────────┘    └─────────┘   └─────────────────┘   └──────────────────┘
```

## Results

<table>
<tr>
<td>

**Overall Metrics**

| Metric | Score |
|--------|-------|
| Accuracy | **93.95%** |
| Precision | 94.28% |
| Recall | 93.95% |
| F1-Score | 93.99% |

</td>
<td>

**Per-Class Performance**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Light_Load | 0.98 | 0.92 | 0.95 |
| Maximum_Load | 0.90 | 0.96 | 0.93 |
| Medium_Load | 0.89 | 0.97 | 0.93 |

</td>
</tr>
</table>

## Project Structure
```
├── Assignment2.ipynb          # Full EDA + model training notebook
├── train_and_save.py          # Script to train & export model artifacts
├── app/
│   ├── main.py                # FastAPI app with web UI
│   ├── model.py               # Model loading & prediction
│   └── feature_engineering.py # Replicates notebook's feature pipeline
├── models/                    # Saved model, scaler, encoder
├── data/load_data.csv         # Dataset
├── Dockerfile                 # Docker build for Render
├── render.yaml                # Render deployment config
└── requirements.txt           # Dependencies
```

## API Usage

**POST** `/predict`
```json
{
  "Date_Time": "01-01-2018 00:15",
  "Usage_kWh": 8.75,
  "Lagging_Current_Reactive_Power_kVarh": 2.95,
  "Leading_Current_Reactive_Power_kVarh": 0.0,
  "CO2_tCO2": 0.0,
  "Lagging_Current_Power_Factor": 73.21,
  "Leading_Current_Power_Factor": 100.0,
  "NSM": 900.0
}
```
**Response:**
```json
{
  "predicted_load_type": "Light_Load",
  "confidence": 0.9953,
  "probabilities": {
    "Light_Load": 0.9953,
    "Maximum_Load": 0.0012,
    "Medium_Load": 0.0035
  }
}
```

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** — web UI with sample data for testing.

## Tech Stack
Python • FastAPI • scikit-learn • XGBoost • LightGBM • Docker • Render
