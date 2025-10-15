import os
import requests
from dotenv import load_dotenv

class EnverusAPI:
    _token: str = ""

    def __init__(self): 
        # Load environment variables from .env file
        load_dotenv()
        self.__api_key = os.getenv("ENVERUS_API_KEY")
        if not self.__api_key:
            raise ValueError("ENVERUS_API_KEY not found in environment variables")
        self.base_url = "https://api.enverus.com/v3/direct-access"

    def generate_token(self) -> None:
        '''Generates an API token.'''
        url = f"{self.base_url}/tokens"
        headers = {"Content-Type": "application/json"}
        data = {"secretKey": self.__api_key}

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        print(response.json().get("token"))

        if not self._token:
            raise ValueError("Failed to retrieve token from response")
        
    