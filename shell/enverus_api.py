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
        '''Generates an API token to use for subsequent requests.'''
        url = f"{self.base_url}/tokens"
        headers = {"Content-Type": "application/json"}
        data = {"secretKey": self.__api_key}

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        self._token = response.json().get("token", "")

        if not self._token:
            raise ValueError("Failed to retrieve token from response")
        
    def query_lmp(self) -> None:
        '''Queries the LMP endpoint using the generated token.'''
        if not self._token:
            raise ValueError("API token is not generated. Call generate_token() first.")
        url = f"{self.base_url}/lmp"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        print(response.json())

if __name__ == "__main__":
    api = EnverusAPI()
    api.generate_token()    
    api.query_lmp()