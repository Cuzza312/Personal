import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Download stock data from Yahoo Finance
ticker = 'AAPL'
df = yf.download(ticker, start='2005-01-01', end='2023-04-27')
tensorflow.random.set_seed(42)

# Preprocess data
scaler = MinMaxScaler()
data = scaler.fit_transform(df[['Close']])
lookback = 2
X = []
y = []
for i in range(lookback, len(data)):
    X.append(data[i-lookback:i, 0])
    y.append(data[i, 0])
X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

y_test1 = y_test
# Train Lasso regression model to select features
lasso = LassoCV(cv=5)
lasso.fit(X_train[:, :, 0], y_train)
mask = lasso.coef_ != 0

# Build Lasso-LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(lookback, 1), activation='tanh')))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(1))
model.compile(optimizer='nadam', loss='mse')

# Train Lasso-LSTM model
history = model.fit(X_train[:, mask], y_train, validation_data=(X_test[:, mask], y_test), epochs=5, batch_size=16)

# Make predictions on test set
y_pred = model.predict(X_test[:, mask])
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

d_y_test = pd.DataFrame(y_test)
d_y_pred = pd.DataFrame(y_pred)

direction = np.where(d_y_test.shift(-1) > d_y_test, 1, 0)
want_buy = np.where(d_y_pred.shift(-1) > d_y_test, 1, 0)

sell = 0
buy = 0
pSell = 0
pBuy = 0
trueBuy = 0

for i in range(direction.shape[0]):
    if want_buy[i] == 0:
       sell += 1
    else:
        buy += 1
        if direction[i] == 1:
            trueBuy += 1

    if direction[i] == 0:
        pSell += 1
    else:
        pBuy += 1

print(f"Buy: {buy}, Sell: {sell}, Proper Buy: {pBuy}, Proper Sell: {pSell}, True Buy: {trueBuy}")

accuracy = np.mean(direction == want_buy) * 100
print(f'Accuracy: {accuracy:.2f}')

# Evaluate model performance
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100
print(f'RMSE: {rmse:.2f}')
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.2f}')

# Predict tomorrow's closing price
last_data = data[-lookback:]
# last_data = last_data[-lookback:, 0]
last_data = last_data[mask]
last_data = last_data.reshape(1, -1, 1)
tomorrow_pred = model.predict(last_data)
tomorrow_pred = scaler.inverse_transform(tomorrow_pred)
print("Previous Close: ", df["Close"][-1])
print(f'Predicted closing price tomorrow: {tomorrow_pred[0, 0]:.2f}')

