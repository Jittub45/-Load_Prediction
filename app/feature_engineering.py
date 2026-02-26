"""
Feature Engineering — Replicates the exact same transformations from the notebook.
Takes raw 8-field input and produces the 31 features the model expects.
"""

import numpy as np
import pandas as pd


def get_time_period(hour: int) -> int:
    """Classify hour into time period."""
    if 6 <= hour < 12:
        return 0    # Morning
    elif 12 <= hour < 18:
        return 1    # Afternoon
    elif 18 <= hour < 22:
        return 2    # Evening
    else:
        return 3    # Night


def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes a dict with the 8 raw input fields and returns a DataFrame
    with the 31 engineered features in the correct order.

    Expected raw keys:
        Date_Time, Usage_kWh, Lagging_Current_Reactive_Power_kVarh,
        Leading_Current_Reactive_Power_kVarh, CO2_tCO2,
        Lagging_Current_Power_Factor, Leading_Current_Power_Factor, NSM
    """

    # --- Map to internal column names (match training data) ---
    dt = pd.to_datetime(raw["Date_Time"])
    usage = float(raw["Usage_kWh"])
    lag_reactive = float(raw["Lagging_Current_Reactive_Power_kVarh"])
    lead_reactive = float(raw["Leading_Current_Reactive_Power_kVarh"])
    co2 = float(raw["CO2_tCO2"])
    lag_pf = float(raw["Lagging_Current_Power_Factor"])
    lead_pf = float(raw["Leading_Current_Power_Factor"])
    nsm = float(raw["NSM"])

    # --- Time Features ---
    hour = dt.hour
    day_of_week = dt.dayofweek
    day_of_month = dt.day
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    time_period = get_time_period(hour)

    # --- Power Features ---
    power_factor_diff = lag_pf - lead_pf
    usage_rate = usage / (nsm + 1)
    reactive_power_ratio = lag_reactive / (lead_reactive + 0.001)

    # --- Cyclical Encoding ---
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # --- Interaction Features ---
    total_reactive_power = lag_reactive + lead_reactive
    reactive_power_diff = lag_reactive - lead_reactive
    usage_squared = usage ** 2
    usage_log = np.log1p(usage)
    power_factor_product = lag_pf * lead_pf
    avg_power_factor = (lag_pf + lead_pf) / 2
    nsm_normalized = nsm / 86400
    usage_x_lag_pf = usage * lag_pf
    usage_x_lead_pf = usage * lead_pf

    # --- Build feature dict in EXACT training column order ---
    features = {
        "Usage_kWh": usage,
        "Lagging_Current_Reactive.Power_kVarh": lag_reactive,
        "Leading_Current_Reactive_Power_kVarh": lead_reactive,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lag_pf,
        "Leading_Current_Power_Factor": lead_pf,
        "NSM": nsm,
        "Hour": hour,
        "Day_of_Week": day_of_week,
        "Day_of_Month": day_of_month,
        "Month": month,
        "Is_Weekend": is_weekend,
        "Time_Period": time_period,
        "Power_Factor_Diff": power_factor_diff,
        "Usage_Rate": usage_rate,
        "Reactive_Power_Ratio": reactive_power_ratio,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
        "DOW_sin": dow_sin,
        "DOW_cos": dow_cos,
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Total_Reactive_Power": total_reactive_power,
        "Reactive_Power_Diff": reactive_power_diff,
        "Usage_kWh_squared": usage_squared,
        "Usage_kWh_log": usage_log,
        "Power_Factor_Product": power_factor_product,
        "Avg_Power_Factor": avg_power_factor,
        "NSM_normalized": nsm_normalized,
        "Usage_x_LagPF": usage_x_lag_pf,
        "Usage_x_LeadPF": usage_x_lead_pf,
    }

    df = pd.DataFrame([features])

    # Replace any inf/nan with 0 (safety net)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df
