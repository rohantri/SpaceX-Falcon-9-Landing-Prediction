import requests
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# load the stock data from the URL
url = "https://ticker.finology.in/api/company/TCS/price_history"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data["price_history"], columns=["Date", "Close"])

# convert the date column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# plot the stock prices
plt.plot(df['Close'])
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.title('TCS Stock Price')
plt.show()

# decompose the time series into trend, seasonality and residuals
decomposition = sm.tsa.seasonal_decompose(df['Close'], model='multiplicative')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# fit a SARIMA model to the residuals
mod = sm.tsa.statespace.SARIMAX(residual,
                                order=(1, 0, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

# predict the next 12 months of stock prices
pred = results.get_prediction(start=pd.to_datetime('2022-01-01'), dynamic=False)
pred_conf = pred.conf_int()

# plot the predicted stock prices along with the confidence interval
ax = df['Close'].plot(label='Observed', figsize=(20, 15))
pred.predicted_mean.plot(ax=ax, label='One-step Ahead Prediction', alpha=.7)
ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Year')
ax.set_ylabel('Stock Price')
plt.legend()
plt.show()
