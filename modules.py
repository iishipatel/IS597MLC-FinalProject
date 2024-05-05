import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file_path):

    traffic_data = pd.read_csv(file_path)

    print(traffic_data.head())

    traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])
    missing_values = traffic_data.isnull().sum()
    statistics = traffic_data.describe()

    print('Missing Values: ', missing_values)
    print('Statistics: ', statistics)

    return traffic_data

def detect_outliers(traffic_data):

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=traffic_data['Vehicles'])
    plt.title('Boxplot of Vehicles Count')
    plt.show()

    # Calculate the 95th percentile for Vehicles
    percentile_95 = traffic_data['Vehicles'].quantile(0.95)
    print('95th Percentile for Vehicles:', percentile_95)

    traffic_data['Vehicles'] = traffic_data['Vehicles'].clip(upper=percentile_95)
    return traffic_data

def preprocess_data(traffic_data):
    scaler = MinMaxScaler()
    traffic_data['Vehicles_normalized'] = scaler.fit_transform(traffic_data[['Vehicles']])

    # Feature Engineering: Extract time-related features from 'DateTime'
    traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
    traffic_data['DayOfWeek'] = traffic_data['DateTime'].dt.dayofweek
    traffic_data['Month'] = traffic_data['DateTime'].dt.month

    # Encode the 'Junction' column as categorical using one-hot encoding
    traffic_data = pd.get_dummies(traffic_data, columns=['Junction'], prefix='Junction')
    junction1_data = traffic_data[traffic_data['Junction_1'] == 1]

    print(junction1_data.head())
    return junction1_data

def arima_specific_preprocessing(junction1_data):
    junction1_series = junction1_data.set_index('DateTime')['Vehicles']
    return junction1_series

def arima_parameter_selection(junction1_series):
    # Perform a Dickey-Fuller test to check stationarity
    result = adfuller(junction1_series.dropna())
    adf_statistic, p_value, usedlag, nobs, critical_values, icbest = result

    print('ADF Statistic Value:', adf_statistic)
    print('P - Value:', p_value)
    print('Critical Values: ', critical_values)

    # Plotting ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    plot_acf(junction1_series, lags=40, ax=ax1)
    ax1.set_title('Autocorrelation Function')

    plot_pacf(junction1_series, lags=40, ax=ax2, method='ywm')
    ax2.set_title('Partial Autocorrelation Function')

    plt.show()

def split_data_arima(junction1_series):
    split_point = int(len(junction1_series) * 0.8)
    train, test = junction1_series[:split_point], junction1_series[split_point:]
    return train, test

def arima_modelling(train, test, p, d, q):

    model_train = ARIMA(train, order=(p, d, q))
    fitted_model_train = model_train.fit()

    # Forecast on the test set
    forecast = fitted_model_train.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Plotting Predicted Trend
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data', color='blue')
    plt.plot(test.index, test, label='Actual Test Data', color='green')
    plt.plot(forecast_mean.index, forecast_mean, label='Predicted Test Data', color='red')
    plt.title('ARIMA Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Vehicles')
    plt.legend()
    plt.show()

    return forecast_mean

def evaluate_arima(test, forecast_mean):

    y_test = np.array(test)  # Actual values
    y_pred = np.array(forecast_mean)  # Predicted values

    # Calculate error metrics
    mae = mean_absolute_error(y_test , y_pred)
    mse = mean_squared_error(y_test , y_pred)
    rmse = np.sqrt(mse)

    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)

def lstm_specific_preprocessing(junction1_data):

    traffic_data = junction1_data[['DateTime', 'Vehicles']]
    traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])
    traffic_data.sort_values('DateTime', inplace=True)
    traffic_data = traffic_data.groupby('DateTime').agg({'Vehicles': 'sum'}).reset_index()

    scaler = MinMaxScaler(feature_range=(0, 1))
    traffic_data['Vehicles'] = scaler.fit_transform(traffic_data[['Vehicles']])

    def create_sequences(data, sequence_length=24):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            xs.append(data[i:(i + sequence_length)])
            ys.append(data[i + sequence_length])
        return np.array(xs), np.array(ys)


    sequence_length = 24  # Using the last 24 hours to predict the next hour
    X, y = create_sequences(traffic_data['Vehicles'], sequence_length)

    return traffic_data, X, y, scaler

def split_data_lstm(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test

def lstm_modelling(X_train, y_train):

    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=64)

    return model, history

def evaluate_lstm(model, traffic_data, X_test, y_train, y_test, scaler):
    
    y_pred = model.predict(X_test)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler.inverse_transform(y_pred).flatten()

    # Calculate error metrics
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(14, 7))
    plt.plot(traffic_data['DateTime'][:len(y_train)], y_train, label='Training Data', color='blue')
    plt.plot(traffic_data['DateTime'][len(y_train):(len(y_train) + len(y_test))], y_test_real, label='Actual Test Data', color='green')
    plt.plot(traffic_data['DateTime'][len(y_train):(len(y_train) + len(y_test))], y_pred_real, label='Predicted Test Data', linestyle='--', color='red')
    plt.title('Traffic Volume Prediction with LSTM')
    plt.xlabel('DateTime')
    plt.ylabel('Vehicles')
    plt.legend()
    plt.show()

    print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')



















