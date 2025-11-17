from enverus_api import EnverusAPI
from market_loads_api import ISODemandController
from load_sarimax_projections import SARIMAXLoadProjections

enverus_api = EnverusAPI()
enverus_api.generate_token()

historic_load_api = ISODemandController("2023-01-01", "2023-01-31", "ERCOT")
historic_loads = historic_load_api.get_market_loads()

sarimax_projections = SARIMAXLoadProjections(historic_loads)