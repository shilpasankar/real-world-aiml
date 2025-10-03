import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Simulate daily passenger count data for 2 years
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Base passenger count with trend and seasonality
trend = np.linspace(1000, 1500, n)  # upward trend
seasonality = 100 * np.sin(2 * np.pi * dates.dayofyear / 365.25)  # yearly seasonality
noise = np.random.normal(0, 50, n)

passenger_counts = trend + seasonality + noise
passenger_counts = np.maximum(passenger_counts, 0).round()  # no negative counts

# Step 2: Prepare DataFrame for Prophet
df = pd.DataFrame({
    'ds': dates,
    'y': passenger_counts
})

# Step 3: Train/Test Split
train_df = df.iloc[:-90]  # last 3 months for testing
test_df = df.iloc[-90:]

# Step 4: Fit Prophet Model
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(train_df)

# Step 5: Make future dataframe and predict
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Step 6: Evaluate model on test data
test_pred = forecast.set_index('ds').loc[test_df['ds'], 'yhat'].values
mae = mean_absolute_error(test_df['y'], test_pred)
rmse = mean_squared_error(test_df['y'], test_pred, squared=False)

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")

# Step 7: Plot results
fig1 = model.plot(forecast)
plt.title("Passenger Count Forecast")
plt.xlabel("Date")
plt.ylabel("Passenger Count")

fig2 = model.plot_components(forecast)
plt.show()
