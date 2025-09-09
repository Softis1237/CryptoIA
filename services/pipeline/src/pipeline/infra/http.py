from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    retries = int(os.getenv("HTTP_RETRIES", "2"))
    backoff = float(os.getenv("HTTP_BACKOFF", "0.5"))
    status_forcelist = (429, 500, 502, 503, 504)
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=status_forcelist, allowed_methods=("GET", "POST"))
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    _SESSION = s
    return s


def http_get(url: str, **kwargs) -> requests.Response:
    if os.getenv("USE_HTTP_RETRY", "1") in {"1", "true", "True"}:
        return _get_session().get(url, **kwargs)
    import requests as _r
    return _r.get(url, **kwargs)


def http_post(url: str, **kwargs) -> requests.Response:
    if os.getenv("USE_HTTP_RETRY", "1") in {"1", "true", "True"}:
        return _get_session().post(url, **kwargs)
    import requests as _r
    return _r.post(url, **kwargs)

