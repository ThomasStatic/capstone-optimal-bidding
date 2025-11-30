from typing import Any
from pandas import DataFrame, Series
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt

class SARIMAXLoadProjections:
    _historic_data: DataFrame
    _y: Any
    y_trans: Any
    X_daily: DataFrame
    exogenous_df: DataFrame
    
    def __init__(self, historic_data: DataFrame, hours: int):
        self._historic_data = historic_data
        self._prepare_historical_data()

        # Because we are working with assumption that the weekly seasonality is stronger
        # than the daily seasonality in load data, we only include Fourier terms for daily seasonality.
        self._set_fourier_terms(self._y.index, period=24, K=3)
    
        self._set_exogenous_df()

        self._fit_sarimax_model(steps = hours)


    def _prepare_historical_data(self):
        '''Prepare historical load data for SARIMAX modeling.'''
        # Convert 'period' to UTC and remove timezone info
        self._y = (self._historic_data.assign(period=lambda x: pd.to_datetime(x['period']).dt.tz_convert("UTC").dt.tz_localize(None))
                .set_index('period').sort_index()["load_MWH"].asfreq('h'))
        
        self._y = pd.to_numeric(self._y, errors='coerce')
        # Fill in missing time periods by interpolating values
        self._y = self._y.interpolate(limit_direction='both')

        # Apply a log transformation to stabilize variance
        self.y_trans = Series(np.log(self._y.clip(lower=1)), index=self._y.index, name="load_MWH_log")

    def _set_fourier_terms(self, index, period, K):
        '''Generate Fourier terms for seasonal components.'''
        t = np.arange(len(index))
        
        X = {f'sin_{period}_{k}': np.sin(2 * np.pi * k * t / period) for k in range(1, K + 1)}
        X |= {f'cos_{period}_{k}': np.cos(2 * np.pi * k * t / period) for k in range(1, K + 1)}
        self.X_daily = DataFrame(X, index=index)

    def _set_exogenous_df(self):
        '''Add calendar features to the exogenous dataframe.'''
        calendar_df = DataFrame(index=self.y_trans.index)
        date_index = pd.DatetimeIndex(calendar_df.index)
        calendar_df['dow'] = date_index.dayofweek
        calendar_df['hour'] = date_index.hour
        
        calendar_df = pd.get_dummies(calendar_df, columns=['dow', 'hour'], drop_first=True)

        self.exogenous_df = pd.concat([self.X_daily, calendar_df], axis=1)

    def _fit_sarimax_model(self, steps: int):
        '''Fit SARIMAX model to the transformed load data.'''
        train_end = self.y_trans.index.max() - pd.Timedelta(days=14)
        y_tr, y_te = self.y_trans[:train_end], self.y_trans[train_end + pd.Timedelta(hours=1):]
        
        exog_tr, exog_te = self.exogenous_df.loc[y_tr.index], self.exogenous_df.loc[y_te.index]
        exog_tr = exog_tr.astype(float) # Ensure exogenous variables are float type as numpy causes errors otherwise
        exog_te = exog_te.astype(float)

        model = SARIMAX(
            y_tr,
            exog=exog_tr,
            order=(1,1,1),
            seasonal_order=(1,1,1,24), # 24 = daily seasonality for hourly data
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        print("Fitting SARIMAX model... this may take a while.")
        fitted_model: Any = model.fit(disp=False) # typed as Any to avoid interpretter issues (statsmodels types are not always well defined)
        print("SARIMAX model fitted.")
        print(fitted_model.summary())

        #TODO: move plotting to analysis directory, i.e., new file
        #fitted_model.plot_diagnostics(figsize=(10, 16))
        #plt.show()

        # Forecast the next 'steps' periods
        exog_future = exog_te.iloc[:steps].astype(float) # we only need exogenous variables for the forecast horizon

        forecast = fitted_model.get_forecast(steps=steps, exog=exog_future)
        forcast_ci = forecast.conf_int()

        # This is on the log scale, i.e., y_tr = log(load_MWH)
        df_forecast_log = pd.DataFrame({
            'forecast_log': forecast.predicted_mean,
            'lower_ci_log': forcast_ci.iloc[:, 0],
            'upper_ci_log': forcast_ci.iloc[:, 1]
        })

        # Transform back to original scale
        df_forecast = pd.DataFrame({
            'forecast': np.exp(df_forecast_log['forecast_log']),
            'lower_ci': np.exp(df_forecast_log['lower_ci_log']),
            'upper_ci': np.exp(df_forecast_log['upper_ci_log'])
        }, index=df_forecast_log.index)

        df_forecast = df_forecast.reset_index().rename(columns={'index': 'datetime'})

        self.forecast_df = df_forecast

        print("Forecast for the next 24 hours:")
        print(df_forecast)

    def get_forecast_df(self) -> DataFrame:
        if not hasattr(self, 'forecast_df'):
            raise ValueError("Forecast has not been generated yet. Call _fit_sarimax_model() first.")
        return self.forecast_df