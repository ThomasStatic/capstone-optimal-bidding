from typing import Any
from pandas import DataFrame, Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXLoadProjections:
    _historic_data: DataFrame
    _y: Series[Any]

    def __init__(self, historic_data: DataFrame):
        self._historic_data = historic_data

    def _prepare_historical_data(self):
        '''Prepare historical load data for SARIMAX modeling.'''
        # Convert 'period' to UTC and remove timezone info
        self._y = (self._historic_data.assign(period=lambda x: x['period'].dt.tz_convert("UTC").dt.tz_localize(None))
                .set_index('period')["load_MWH"].asfreq('H'))
        
        # Fill in missing time periods by interpolating values
        self._y = self._y.interpolate(limit_direction='both')