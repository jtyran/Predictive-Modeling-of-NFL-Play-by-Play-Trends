# Predictive Modeling of NFL Play by Play Trends

## Project Overview
This project applies time series forecasting models to NFL play-by-play data to predict future scoring patterns and momentum trends. Using weekly aggregated statistics from 2014 to 2024, the analysis explores short-term forecasting of points scored across games and identifies underlying patterns in team performance.

## Objectives
- Develop an ARIMA-based forecasting model to predict weekly points scored.
- Decompose time series into trend and seasonality components.
- Evaluate model performance through visualization of actual vs. predicted outcomes.
- Analyze the predictive quality of the model for future NFL games.

## Dataset
- **Source:** NFL Play-by-Play data (2014–2024)
- **Frequency:** Weekly aggregation
- **Key Metric:** Points Scored

## Methods
- Time series decomposition
- Train/Test split respecting chronological order
- ARIMA modeling
- Forecast visualization and model diagnostics

## Results
- Successfully captured key trend and seasonal patterns in points scored.
- Forecasts generated for the 2024 NFL season.
- Model components visualized to interpret trend and seasonality effects.

## How to View the Project
- [View the Full Project Report (HTML)](https://jtyran.github.io/Predictive-Modeling-of-NFL-Play-by-Play-Trends/)

---

# Notes
✅ *Train/Test Split:*  
Training data includes games up to **January 2024**, and testing data includes games starting from **August 2024** onward.

✅ *Forecasting Horizon:*  
Primarily focuses on the regular season of the 2024 NFL season.

✅ *Tools Used:*  
Python (Pandas, Statsmodels, Matplotlib)

---

# Future Work
- Experiment with SARIMA or Prophet models for enhanced seasonal prediction.
- Incorporate additional features such as player statistics, weather, or betting lines.
- Expand evaluation metrics to RMSE, MAE, and coverage of prediction intervals.
