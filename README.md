⚡ City Power Consumption Forecasting System
Hourly Forecasting Using Machine Learning + Streamlit Dashboard
<img src="history_vs_future.png" width="600">
📌 Overview

This project predicts future hourly power consumption for a city using historical energy usage data.
It uses:

Time-series feature engineering

A Random Forest Regression model

Recursive multi-step forecasting

A full Streamlit web dashboard

Clean data pipeline + saved artifacts

The dataset used is the UCI Individual Household Electric Power Consumption dataset, which provides minute-level electricity consumption data from 2006–2010.
It is aggregated to hourly values to make forecasts more stable.

⭐ Key Features

✔️ Load & clean raw UCI dataset
✔️ Resample minute-level readings to hourly averages
✔️ Feature-engineering:

24 lag features

Rolling means (3h, 6h, 12h, 24h)

Calendar features (hour/day/month/weekday/weekend)
✔️ Train/test split + full evaluation
✔️ Save trained model + feature names + last known state
✔️ Forecast any number of future hours (recursive)
✔️ Visualize forecasts vs historical data
✔️ Full Streamlit dashboard
✔️ Download CSV_forecast outputs

📊 Model Performance

After adding rolling features, lag features, and calendar variables:

MAE ≈ 0.034 kW
RMSE ≈ 0.061 kW

This is extremely accurate for a real-world forecasting problem.

🧠 Machine Learning Model

Algorithm: RandomForestRegressor

300 Trees

Max depth: 20

Random state: 42

Trained on all engineered features

Very stable & handles nonlinear patterns

📁 Project Structure
project/
│── app.py                     # Streamlit dashboard
│── power.py                   # CLI forecasting script
│── rf_power_model.joblib      # Saved model
│── feature_cols.json          # Required feature order
│── last_state.csv             # Model's final feature row
│── last_time.txt              # Last historical timestamp
│── forecast_next_7_days.csv   # Example forecast output
│── history_vs_future.png      # Visualization
│── requirements.txt
│── README.md

📥 Dataset

Source:
UCI Machine Learning Repository –
“Individual Household Electric Power Consumption”
🔗 https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Download the household_power_consumption.txt file and place it in the project folder.

🛠 Installation and Setup
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run app.py

3️⃣ Use CLI to generate forecasts
python3 power.py --hours 168 --out forecast.csv

🖥 Streamlit Dashboard Features
Mode 1 — Train from dataset

Upload the UCI .txt file

Clean + preprocess data

Train RandomForest model

Save artifacts

View forecast & download CSV

Mode 2 — Use saved model

Instantly forecast any number of hours

Visualize results

Overlay historical actuals

Download output

📈 Forecast Example
<img src="28d90be3-947c-4dbb-aa2e-c8e04d699678.png" width="600">

Blue = actual last 7 days

Orange = forecast next 7 days

Clear daily cycles appearing in forecast

Smooth curve due to model stability

🔮 Future Improvements

Add weather data to improve accuracy further

Integrate LSTM / GRU deep learning models

Add scenario forecasting (hot day, holiday effect, etc.)

Build monthly/weekly summary reports

Deploy online via Streamlit Cloud / AWS / Heroku

💡 Why This Project Is Useful

Energy demand forecasting is critical for:

Smart grid systems

City infrastructure planning

Power load management

Reducing outages

Predicting peak hours

This project provides a fully functional, modular, and extendable forecasting solution.
