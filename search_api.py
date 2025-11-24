# tool for finding comapny
import os
import requests


class SerperSearchTool:
    """
    A reusable tool class used by multiple agents.
    """

    ENDPOINT = "https://google.serper.dev/search"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError("❌ SERPER_API_KEY missing in .env")

    def search(self, query: str, num_results: int = 5):
        payload = {"q": query, "num": num_results}
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(self.ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data.get("organic", [])
