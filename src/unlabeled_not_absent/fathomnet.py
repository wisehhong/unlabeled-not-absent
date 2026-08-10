"""Small, dependency-free client for the public FathomNet image endpoint."""

from __future__ import annotations

import hashlib
import json
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

API_URL = "https://database.fathomnet.org/api/images"
USER_AGENT = "unlabeled-not-absent-poc/0.1"


class FathomNetError(RuntimeError):
    """Raised when FathomNet data cannot be fetched or validated."""


def _request(url: str, *, timeout: float = 30, attempts: int = 3) -> bytes:
    try:
        import certifi
    except ImportError as exc:
        raise FathomNetError(
            "The certifi package is required for verified HTTPS requests; install the project dependencies."
        ) from exc
    tls_context = ssl.create_default_context(cafile=certifi.where())
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout, context=tls_context) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise FathomNetError(f"Request failed after {attempts} attempts: {url}") from exc
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")

# fetch image metadata from the FathomNet API, returning a dict with keys like 'uuid', 'url', 
# 'width', 'height', and 'boundingBoxes'.
def fetch_image_page(page: int, size: int, *, api_url: str = API_URL) -> dict[str, Any]:
    if page < 0:
        raise ValueError("page must be non-negative")
    if not 1 <= size <= 100:
        raise ValueError("size must be between 1 and 100")
    url = f"{api_url}?{urlencode({'page': page, 'size': size})}"
    try:
        payload = json.loads(_request(url))
    except json.JSONDecodeError as exc:
        raise FathomNetError(f"FathomNet returned invalid JSON for page {page}") from exc
    if not isinstance(payload.get("content"), list):
        raise FathomNetError("FathomNet response is missing a content list")
    return payload


def image_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_image(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(_request(url, timeout=90))
    if expected_sha256 and file_sha256(temporary) != expected_sha256.lower():
        temporary.unlink()
        raise FathomNetError(f"Downloaded image failed SHA-256 validation: {url}")
    temporary.replace(destination)
