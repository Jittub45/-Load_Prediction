"""
FastAPI application for Load Type prediction.
Provides both a REST API and a web-based UI.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from app.model import load_artifacts, predict
from app.feature_engineering import engineer_features


# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Load Type Predictor",
    description="Predicts power system Load Type (Light / Medium / Maximum) from energy consumption data.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic schema ---
class PredictionInput(BaseModel):
    Date_Time: str = Field(..., example="01-01-2018 00:15", description="Timestamp (DD-MM-YYYY HH:MM)")
    Usage_kWh: float = Field(..., example=8.75, description="Energy consumption in kWh")
    Lagging_Current_Reactive_Power_kVarh: float = Field(..., example=2.95, description="Lagging current reactive power (kVarh)")
    Leading_Current_Reactive_Power_kVarh: float = Field(..., example=0.0, description="Leading current reactive power (kVarh)")
    CO2_tCO2: float = Field(..., example=0.0, description="CO2 emissions (tCO2)")
    Lagging_Current_Power_Factor: float = Field(..., example=73.21, description="Lagging current power factor (%)")
    Leading_Current_Power_Factor: float = Field(..., example=100.0, description="Leading current power factor (%)")
    NSM: float = Field(..., example=900.0, description="Number of seconds from midnight")


class PredictionOutput(BaseModel):
    predicted_load_type: str
    confidence: float
    probabilities: dict


# --- API Routes ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "Stacking Ensemble"}


@app.post("/predict", response_model=PredictionOutput)
def predict_load_type(input_data: PredictionInput):
    """Predict load type from raw input features."""
    raw = input_data.model_dump()
    features_df = engineer_features(raw)
    result = predict(features_df)
    return result


# --- Web UI ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Load Type Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .page-wrapper {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 24px;
            width: 100%;
            max-width: 1060px;
            align-items: start;
        }
        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        }
        h1 {
            color: #fff;
            text-align: center;
            margin-bottom: 8px;
            font-size: 28px;
            font-weight: 700;
        }
        .subtitle {
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            color: rgba(255, 255, 255, 0.8);
            font-size: 13px;
            margin-bottom: 6px;
            font-weight: 500;
        }
        label .hint {
            color: rgba(255, 255, 255, 0.4);
            font-weight: 400;
        }
        input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            color: #fff;
            font-size: 15px;
            transition: all 0.3s;
            outline: none;
        }
        input:focus {
            border-color: #6c63ff;
            background: rgba(255, 255, 255, 0.12);
            box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.2);
        }
        input::placeholder { color: rgba(255, 255, 255, 0.3); }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #6c63ff, #3f51b5);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: all 0.3s;
            letter-spacing: 0.5px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(108, 99, 255, 0.4);
        }
        .btn:active { transform: translateY(0); }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 24px;
            padding: 24px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: none;
        }
        .result.show { display: block; animation: fadeIn 0.4s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .result-label {
            color: rgba(255, 255, 255, 0.5);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .result-value {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 16px;
        }
        .light { color: #2ecc71; }
        .medium { color: #f39c12; }
        .maximum { color: #e74c3c; }
        .confidence {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-bottom: 16px;
        }
        .prob-bar-container {
            margin-bottom: 8px;
        }
        .prob-label {
            display: flex;
            justify-content: space-between;
            color: rgba(255, 255, 255, 0.7);
            font-size: 13px;
            margin-bottom: 4px;
        }
        .prob-bar {
            height: 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s ease;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 16px;
            font-size: 14px;
        }
        .sidebar {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            position: sticky;
            top: 20px;
        }
        .samples-title {
            color: rgba(255,255,255,0.7);
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
            text-align: center;
        }
        .sample-cards {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .sample-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 14px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }
        .sample-card:hover {
            background: rgba(255,255,255,0.12);
            border-color: rgba(255,255,255,0.25);
            transform: translateY(-2px);
        }
        @media (max-width: 800px) {
            .page-wrapper {
                grid-template-columns: 1fr;
            }
            .sidebar { order: -1; }
        }
        .sample-card .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .sample-card .badge.light { background: rgba(46,204,113,0.2); color: #2ecc71; }
        .sample-card .badge.medium { background: rgba(243,156,18,0.2); color: #f39c12; }
        .sample-card .badge.maximum { background: rgba(231,76,60,0.2); color: #e74c3c; }
        .sample-card .sample-detail {
            color: rgba(255,255,255,0.5);
            font-size: 10px;
            line-height: 1.6;
            text-align: left;
        }
        .sample-card .sample-detail span {
            color: rgba(255,255,255,0.8);
            font-weight: 500;
        }
        .sample-card .try-btn {
            display: inline-block;
            margin-top: 8px;
            padding: 4px 14px;
            background: rgba(108,99,255,0.3);
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .sample-card .try-btn:hover {
            background: rgba(108,99,255,0.6);
        }
    </style>
</head>
<body>
    <div class="page-wrapper">
    <div class="container">
        <h1>&#9889; Load Type Predictor</h1>
        <p class="subtitle">Predict power system load type from energy consumption data</p>

        <form id="predForm">
            <div class="form-group">
                <label>Date & Time <span class="hint">(DD-MM-YYYY HH:MM)</span></label>
                <input type="text" id="Date_Time" placeholder="01-01-2018 00:15" value="01-01-2018 00:15" required>
            </div>

            <div class="form-group">
                <label>Usage_kWh <span class="hint">(Industry Energy Consumption in kWh)</span></label>
                <input type="number" id="Usage_kWh" step="any" placeholder="8.75" required>
            </div>

            <div class="form-group">
                <label>Lagging_Current_Reactive_Power_kVarh</label>
                <input type="number" id="Lagging_Current_Reactive_Power_kVarh" step="any" placeholder="2.95" required>
            </div>

            <div class="form-group">
                <label>Leading_Current_Reactive_Power_kVarh</label>
                <input type="number" id="Leading_Current_Reactive_Power_kVarh" step="any" placeholder="0.0" required>
            </div>

            <div class="form-group">
                <label>CO2_tCO2 <span class="hint">(CO2 Emissions in tCO2)</span></label>
                <input type="number" id="CO2_tCO2" step="any" placeholder="0.0" required>
            </div>

            <div class="form-group">
                <label>Lagging_Current_Power_Factor <span class="hint">(%)</span></label>
                <input type="number" id="Lagging_Current_Power_Factor" step="any" placeholder="73.21" required>
            </div>

            <div class="form-group">
                <label>Leading_Current_Power_Factor <span class="hint">(%)</span></label>
                <input type="number" id="Leading_Current_Power_Factor" step="any" placeholder="100.0" required>
            </div>

            <div class="form-group">
                <label>NSM <span class="hint">(Number of Seconds from Midnight)</span></label>
                <input type="number" id="NSM" step="any" placeholder="900" required>
            </div>

            <button type="submit" class="btn" id="submitBtn">Predict Load Type</button>
        </form>

        <div class="error" id="errorMsg"></div>

        <div class="result" id="resultBox">
            <div class="result-label">Predicted Load Type</div>
            <div class="result-value" id="predLabel"></div>
            <div class="confidence" id="confidence"></div>

            <div class="prob-bar-container">
                <div class="prob-label"><span>Light Load</span><span id="probLight">0%</span></div>
                <div class="prob-bar"><div class="prob-bar-fill" id="barLight" style="width:0%;background:#2ecc71;"></div></div>
            </div>
            <div class="prob-bar-container">
                <div class="prob-label"><span>Medium Load</span><span id="probMedium">0%</span></div>
                <div class="prob-bar"><div class="prob-bar-fill" id="barMedium" style="width:0%;background:#f39c12;"></div></div>
            </div>
            <div class="prob-bar-container">
                <div class="prob-label"><span>Maximum Load</span><span id="probMax">0%</span></div>
                <div class="prob-bar"><div class="prob-bar-fill" id="barMax" style="width:0%;background:#e74c3c;"></div></div>
            </div>
        </div>
    </div>

    <!-- Sidebar: Sample Data -->
    <div class="sidebar">
        <div class="samples-title">Sample Data from CSV<br><small style="font-weight:400;font-size:10px;letter-spacing:0;text-transform:none;color:rgba(255,255,255,0.4);">Click to auto-fill the form</small></div>
        <div class="sample-cards">
            <div class="sample-card" onclick="fillSample('light')">
                <div class="badge light">Light Load</div>
                <div class="sample-detail">
                    Date_Time: <span>01-01-2018 00:15</span><br>
                    Usage_kWh: <span>8.75</span><br>
                    Lag_React_Power: <span>2.95</span><br>
                    Lead_React_Power: <span>0.0</span><br>
                    CO2: <span>0.0</span><br>
                    Lag_Power_Factor: <span>73.21</span><br>
                    Lead_Power_Factor: <span>100.0</span><br>
                    NSM: <span>900</span>
                </div>
                <button class="try-btn" type="button">Try This</button>
            </div>
            <div class="sample-card" onclick="fillSample('medium')">
                <div class="badge medium">Medium Load</div>
                <div class="sample-detail">
                    Date_Time: <span>02-01-2018 09:15</span><br>
                    Usage_kWh: <span>56.84</span><br>
                    Lag_React_Power: <span>8.32</span><br>
                    Lead_React_Power: <span>0.0</span><br>
                    CO2: <span>0.0</span><br>
                    Lag_Power_Factor: <span>151.62</span><br>
                    Lead_Power_Factor: <span>100.0</span><br>
                    NSM: <span>33300</span>
                </div>
                <button class="try-btn" type="button">Try This</button>
            </div>
            <div class="sample-card" onclick="fillSample('maximum')">
                <div class="badge maximum">Maximum Load</div>
                <div class="sample-detail">
                    Date_Time: <span>02-01-2018 10:15</span><br>
                    Usage_kWh: <span>54.79</span><br>
                    Lag_React_Power: <span>7.52</span><br>
                    Lead_React_Power: <span>0.35</span><br>
                    CO2: <span>0.0</span><br>
                    Lag_Power_Factor: <span>99.07</span><br>
                    Lead_Power_Factor: <span>100.0</span><br>
                    NSM: <span>36900</span>
                </div>
                <button class="try-btn" type="button">Try This</button>
            </div>
        </div>
    </div>
    </div><!-- /page-wrapper -->

    <script>
        const sampleData = {
            light: {
                Date_Time: '01-01-2018 00:15',
                Usage_kWh: 8.753692425450835,
                Lagging_Current_Reactive_Power_kVarh: 2.95,
                Leading_Current_Reactive_Power_kVarh: 0.0,
                CO2_tCO2: 0.0,
                Lagging_Current_Power_Factor: 73.21,
                Leading_Current_Power_Factor: 100.0,
                NSM: 900.0
            },
            medium: {
                Date_Time: '02-01-2018 09:15',
                Usage_kWh: 56.84,
                Lagging_Current_Reactive_Power_kVarh: 8.32,
                Leading_Current_Reactive_Power_kVarh: 0.0,
                CO2_tCO2: 0.0,
                Lagging_Current_Power_Factor: 151.62108682743553,
                Leading_Current_Power_Factor: 100.0,
                NSM: 33300.0
            },
            maximum: {
                Date_Time: '02-01-2018 10:15',
                Usage_kWh: 54.79,
                Lagging_Current_Reactive_Power_kVarh: 7.52,
                Leading_Current_Reactive_Power_kVarh: 0.3460346233547498,
                CO2_tCO2: 0.0,
                Lagging_Current_Power_Factor: 99.07,
                Leading_Current_Power_Factor: 100.0,
                NSM: 36900.0
            }
        };

        function fillSample(type) {
            const data = sampleData[type];
            for (const [key, val] of Object.entries(data)) {
                document.getElementById(key).value = val;
            }
        }

        document.getElementById('predForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const errEl = document.getElementById('errorMsg');
            const resultBox = document.getElementById('resultBox');
            errEl.textContent = '';
            resultBox.classList.remove('show');
            btn.disabled = true;
            btn.textContent = 'Predicting...';

            const payload = {
                Date_Time: document.getElementById('Date_Time').value,
                Usage_kWh: parseFloat(document.getElementById('Usage_kWh').value),
                Lagging_Current_Reactive_Power_kVarh: parseFloat(document.getElementById('Lagging_Current_Reactive_Power_kVarh').value),
                Leading_Current_Reactive_Power_kVarh: parseFloat(document.getElementById('Leading_Current_Reactive_Power_kVarh').value),
                CO2_tCO2: parseFloat(document.getElementById('CO2_tCO2').value),
                Lagging_Current_Power_Factor: parseFloat(document.getElementById('Lagging_Current_Power_Factor').value),
                Leading_Current_Power_Factor: parseFloat(document.getElementById('Leading_Current_Power_Factor').value),
                NSM: parseFloat(document.getElementById('NSM').value),
            };

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Prediction failed');
                }
                const data = await resp.json();

                // Display result
                const label = data.predicted_load_type;
                const predEl = document.getElementById('predLabel');
                predEl.textContent = label.replace('_', ' ');
                predEl.className = 'result-value';
                if (label.includes('Light')) predEl.classList.add('light');
                else if (label.includes('Medium')) predEl.classList.add('medium');
                else predEl.classList.add('maximum');

                document.getElementById('confidence').textContent =
                    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

                const probs = data.probabilities;
                const lightP = (probs['Light_Load'] || 0) * 100;
                const medP = (probs['Medium_Load'] || 0) * 100;
                const maxP = (probs['Maximum_Load'] || 0) * 100;

                document.getElementById('probLight').textContent = lightP.toFixed(1) + '%';
                document.getElementById('barLight').style.width = lightP + '%';
                document.getElementById('probMedium').textContent = medP.toFixed(1) + '%';
                document.getElementById('barMedium').style.width = medP + '%';
                document.getElementById('probMax').textContent = maxP.toFixed(1) + '%';
                document.getElementById('barMax').style.width = maxP + '%';

                resultBox.classList.add('show');
            } catch (err) {
                errEl.textContent = err.message;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Predict Load Type';
            }
        });
    </script>
</body>
</html>
"""
