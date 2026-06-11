import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

df["Order Date"] = pd.to_datetime(df["Order Date"])

daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()

daily_sales["Day"] = range(len(daily_sales))

X = daily_sales[["Day"]]
y = daily_sales["Sales"]

model = LinearRegression()
model.fit(X, y)

future_days = pd.DataFrame(
    {"Day": range(len(daily_sales), len(daily_sales) + 30)}
)

forecast = model.predict(future_days)

plt.figure(figsize=(10,5))
plt.plot(daily_sales["Order Date"], y, label="Actual Sales")
plt.plot(
    pd.date_range(
        start=daily_sales["Order Date"].max(),
        periods=30,
        freq="D"
    ),
    forecast,
    label="Forecast"
)
plt.legend()
plt.title("Sales Forecast")
plt.savefig("sales_forecast.png")
plt.show()