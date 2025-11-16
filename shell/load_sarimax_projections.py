from typing import Any
from pandas import DataFrame, Series
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

class SARIMAXLoadProjections:
    _historic_data: DataFrame
    _y: Series[Any]
    y_trans: Series[Any]
    X_daily: DataFrame
    exogenous_df: DataFrame
    
    def __init__(self, historic_data: DataFrame):
        self._historic_data = historic_data
        self._prepare_historical_data()

        # Because we are working with assumption that the weekly seasonality is stronger
        # than the daily seasonality in load data, we only include Fourier terms for daily seasonality.
        self.set_fourier_terms(self._y.index, period=24, K=3)

    def _prepare_historical_data(self):
        '''Prepare historical load data for SARIMAX modeling.'''
        # Convert 'period' to UTC and remove timezone info
        self._y = (self._historic_data.assign(period=lambda x: x['period'].dt.tz_convert("UTC").dt.tz_localize(None))
                .set_index('period').sort_index()["load_MWH"].asfreq('h'))
        
        self._y = pd.to_numeric (self._y, errors='coerce')
        # Fill in missing time periods by interpolating values
        self._y = self._y.interpolate(limit_direction='both')

        # Apply a log transformation to stabilize variance
        self.y_trans = Series(np.log(self._y.clip(lower=1)), index=self._y.index, name="load_MWH_log")

    def set_fourier_terms(self, index, period, K):
        '''Generate Fourier terms for seasonal components.'''
        t = np.arange(len(index))
        
        X = {f'sin_{period}_{k}': np.sin(2 * np.pi * k * t / period) for k in range(1, K + 1)}
        X |= {f'cos_{period}_{k}': np.cos(2 * np.pi * k * t / period) for k in range(1, K + 1)}
        self.X_daily = DataFrame(X, index=index)

    def set_exogenous_df(self):
        '''Add calendar features to the exogenous dataframe.'''
        calendar_df = DataFrame(index=self.y_trans.index)
        date_index = pd.DatetimeIndex(calendar_df.index)
        calendar_df['dow'] = date_index.dayofweek
        calendar_df['hour'] = date_index.hour
        
        calendar_df = pd.get_dummies(calendar_df, columns=['dow', 'hour'], drop_first=True)

        self.exogenous_df = pd.concat([self.X_daily, calendar_df], axis=1)
