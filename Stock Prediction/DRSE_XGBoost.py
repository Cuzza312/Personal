import numpy as np
import pandas as pd
import tensorflow.keras.optimizers
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
import yfinance as yf


def run_model():
    # Load the data
    data = pd.read_csv('stock_data.csv', index_col="Date")

    ticker = 'AAPL'
    data = yf.download(ticker, start='2005-01-01', end='2023-06-2')

    data = ta.add_all_ta_features(df=data, high="High", low="Low", close="Close", open="Open", volume="Volume")

    data.drop(["High", "Low", "Open", "Adj Close"], axis=1, inplace=True)

    data.fillna(data.mean(), inplace=True)
    np.random.seed(42)

    latest_data_p = data[-1:]
    data = data.drop(data.tail(1).index)
    print(latest_data_p)

    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scalar.fit_transform(data), columns=data.columns)

    X = scaled_data.drop('Close', axis=1)
    y = scaled_data['Close']

    # Feature selection using mutual information
    # Feature selection using mutual information
    mi = mutual_info_regression(X, y)
    mi2 = pd.Series(mi)
    mi2.index = X.columns
    from sklearn.feature_selection import SelectPercentile

    selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
    selected_top_columns.fit(X.fillna(0), y)
    selected_top_columns.get_support()
    selected_features = X.columns[selected_top_columns.get_support()]

    print(selected_features)

    X_train, X_test, y_train, y_test = train_test_split(X[selected_features].shift(1).dropna(), y.iloc[1:], test_size=0.2, random_state=42, shuffle=True)

    length = int(y_test.shape[0])
    print(length)

    # Define the parameters for the XGBoost model
    params = {
        'max_depth': 10,
        'eta': 0.001,
        'learning_rate': 0.5,
        'booster': 'gbtree',
    }

    num_rounds = 100

    # Build the deep random subspace ensemble model
    n_features = len(selected_features)
    n_subnets = 10
    subnets = []
    for i in range(n_subnets):
        # Random subspace sampling
        subset = np.random.choice(n_features, size=int(n_features*0.6), replace=False)
        X_subset_train = X_train.iloc[:, subset]
        # Train a neural network on the subset of features
        # Convert the training data to an optimized data structure for XGBoost
        dtrain = xgb.DMatrix(X_subset_train, label=y_train)

        # Train the XGBoost model
        xgb_model = xgb.train(params, dtrain, 20)
        subnets.append((xgb_model, subset))

    # Make predictions using the ensemble model
    predictions = np.zeros_like(y_test)
    for clf, subset in subnets:
        X_test_subset = X_test.iloc[:, subset]
        # Convert the test data to an optimized data structure for XGBoost
        dtest = xgb.DMatrix(X_test_subset)

        # Make predictions using the trained XGBoost model
        pred = clf.predict(dtest)
        predictions += pred / n_subnets


    y_test2 = pd.DataFrame(y_test)
    previousCloses = []
    for idx, a in y_test2.iterrows():
        indx = int(idx) - 1
        previousCloses.append(y[indx])

    previousCloses = pd.DataFrame(previousCloses, columns=y_test2.columns).reset_index(drop=True)
    predictions = pd.DataFrame(predictions, columns=y_test2.columns).reset_index(drop=True)
    y_test1 = pd.DataFrame(y_test2, columns=y_test2.columns).reset_index(drop=True)

    y_test1 = y_test1 * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)
    previousCloses = previousCloses * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)
    predictions1 = predictions * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)

    # Direction = (y_test1 > previousCloses).astype(int)

    # want_buy = (predictions1 > previousCloses).astype(int)

    Direction = np.where(y_test1 > previousCloses, 1, 0)
    want_buy = np.where(predictions1 > previousCloses, 1, 0)

    # for i in range(length):
    #    if predictions[i] > y_test[i]:
    #        want_buy.append(1)
    #    else:
    #        want_buy.append(0)

    sell = 0
    buy = 0
    pSell = 0
    pBuy = 0
    trueBuy = 0

    for i in range(Direction.shape[0]):
        if want_buy[i] == 0:
            sell += 1
        else:
            buy += 1
            if Direction[i] == 1:
                trueBuy += 1

        if Direction[i] == 0:
            pSell += 1
        else:
            pBuy += 1

    print(f"Buy: {buy}, Sell: {sell}, Proper Buy: {pBuy}, Proper Sell: {pSell}, True Buy: {trueBuy}")

    want_buy = pd.DataFrame(want_buy, columns=predictions.columns).reset_index(drop=True)
    Direction = pd.DataFrame(Direction, columns=predictions.columns).reset_index(drop=True)

    correct = np.mean(want_buy == Direction) * 100

    score = np.sum(correct)

    print(f"Score: {score}")


    y_test1 = y_test * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)
    predictions1 = predictions * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)

    plotting = False
    if plotting is True:
        x = np.arange(y_test1[-60:].shape[0])
        plt.plot(x, y_test1[-60:], label="Actual")
        plt.plot(x, predictions1[-60:], label="Predicted")
        plt.title('Actual test samples vs. forecasts')
        plt.legend()
        plt.show()

    # predictions = predictions.shift(-1).dropna()
    # y_test = y_test[:-1]

    y_test1 = y_test * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)
    predictions1 = predictions * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)

    # Evaluate the performance of the model
    mse = mean_squared_error(y_test1, predictions1)
    print("MSE: ", mse)

    mse = np.sqrt(mse)
    print("RMSE: ", mse)

    mae = np.sqrt(mean_absolute_error(y_test1, predictions1))
    print("MAE: ", mae)

    r2 = r2_score(y_test1, predictions1)
    print("R2: ", r2)

    print("Accuracy: ", score)

    # Load the latest data
    latest_data = latest_data_p
    X_latest = pd.DataFrame(scalar.transform(latest_data), columns=latest_data.columns)
    X_latest = X_latest[selected_features]

    # Make predictions for tomorrow's close price
    predictions_tomorrow = np.zeros(n_subnets)
    for i, (clf, subset) in enumerate(subnets):
        X_subset_tomorrow = X_latest.iloc[:, subset]
        # Convert the test data to an optimized data structure for XGBoost
        dtest = xgb.DMatrix(X_subset_tomorrow)
        pred = clf.predict(dtest)
        predictions_tomorrow[i] = pred[len(pred) - 1]

    prediction_tomorrow = np.mean(predictions_tomorrow)

    real_yTest = y_test * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)

    prediction_tomorrow = prediction_tomorrow * (data["Close"].max(axis=0) - data["Close"].min(axis=0)) + data["Close"].min(axis=0)
    print("Previous Close: ", latest_data_p["Close"][-1])
    print("Predicted close price for tomorrow: ", prediction_tomorrow)

    prediction_tomorrow = float(prediction_tomorrow)

    return prediction_tomorrow


if __name__ == "__main__":
    run_model()


