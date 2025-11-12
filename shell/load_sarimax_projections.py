from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXLoadProjections:
    def __init__(self, historic_data):
        self.historic_data = historic_data

    