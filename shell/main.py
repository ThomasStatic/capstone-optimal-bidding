from enverus_api import EnverusAPI
from market_loads_api import ISODemandController

#api = EnverusAPI()
#api.generate_token()

marketDemandController = ISODemandController("2023-01-01", "2023-01-31")
marketDemandController.get_market_loads()