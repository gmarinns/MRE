"""Cliente público com timeout, tentativas limitadas e identificação."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def session() -> requests.Session:
    client = requests.Session()
    client.headers["User-Agent"] = "FinProv/0.1 (academic market-risk research)"
    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=False,
    )
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def fetch(client, url: str, params: dict | None = None) -> bytes:
    response = client.get(url, params=params, timeout=(10, 40))
    response.raise_for_status()
    return response.content
