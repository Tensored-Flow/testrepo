"""Client for the internal data API."""

import json
import urllib.request


class DataAPIClient:
    """Client for the internal data API.

    Category B: All methods have network side effects.
    """

    def __init__(self, base_url, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def get_dataset(self, dataset_id):
        """Fetch a dataset by ID."""
        url = f"{self.base_url}/datasets/{dataset_id}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(req)
            return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def submit_results(self, dataset_id, results):
        """Submit processed results back to the API."""
        url = f"{self.base_url}/datasets/{dataset_id}/results"
        data = json.dumps(results).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            response = urllib.request.urlopen(req)
            return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}
