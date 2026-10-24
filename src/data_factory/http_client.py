"""HTTP layer: send a request, retry what is worth retrying, and only return valid JSON.

This module knows nothing about NYC 311, Socrata or the contract. It answers one question:
"give me the JSON at this URL, or tell me precisely why you could not".

Behaviour per outcome (decision D21 in docs/decisions.md):
  - timeout, connection error, 429, 500, 502, 503, 504   -> retry with exponential backoff
  - any other error status (400, 403, 404, ...)          -> HttpStatusError, no retry
  - status 2xx but not JSON content (e.g. an HTML page)  -> InvalidResponseError, no retry
  - JSON content that cannot be parsed (truncated body)  -> retry, then InvalidResponseError
  - text download (get_text) answered with an HTML page   -> InvalidResponseError, no retry
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from data_factory.exceptions import HttpStatusError, InvalidResponseError, NetworkError

log = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
BODY_EXCERPT_CHARS = 300   # how much of an error body is kept in the error message


@dataclass
class RequestStats:
    calls: int = 0      # logical requests asked for by the caller
    attempts: int = 0   # HTTP requests actually sent, retries included
    retries: int = 0


class _Retryable(Exception):
    """Internal signal: this attempt failed in a way that another attempt may fix."""

    def __init__(self, message: str, retry_after: float | None = None, bad_body: bool = False):
        super().__init__(message)
        self.retry_after = retry_after   # seconds the server asked us to wait, if any
        self.bad_body = bad_body         # the server answered, but the body was unusable


class HttpClient:
    def __init__(
        self,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        headers: dict[str, str] | None = None,
    ):
        # `session` and `sleep` can be replaced in tests: no network, no real waiting.
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = session or requests.Session()
        self.sleep = sleep
        self.headers = headers or {}
        self.stats = RequestStats()

    # ------------------------------------------------------------ public API
    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.request_json("GET", url, params=params)

    def post_json(self, url: str, body: Any, params: dict[str, Any] | None = None) -> Any:
        return self.request_json("POST", url, params=params, body=body)

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """A plain-text file (e.g. a pipe-delimited table). An HTML page is refused."""
        return self._request("GET", url, params, None, _parse_text)

    def request_json(self, method: str, url: str, params: dict[str, Any] | None = None,
                     body: Any = None) -> Any:
        return self._request(method, url, params, body, _parse_json)

    def _request(self, method: str, url: str, params: dict[str, Any] | None, body: Any,
                 parse: Callable[[str, str, requests.Response], Any]) -> Any:
        self.stats.calls += 1
        last_problem = _Retryable("no attempt made")
        for attempt in range(1, self.max_retries + 2):
            self.stats.attempts += 1
            try:
                return parse(method, url, self._send(method, url, params, body))
            except _Retryable as problem:
                last_problem = problem
                if attempt > self.max_retries:
                    break
                wait = self._wait_seconds(attempt, problem)
                self.stats.retries += 1
                log.warning("%s %s failed (%s); retry %d/%d in %.1fs",
                            method, url, problem, attempt, self.max_retries, wait)
                self.sleep(wait)
        message = f"{method} {url}: {last_problem} (after {self.max_retries + 1} attempts)"
        if last_problem.bad_body:
            raise InvalidResponseError(message)
        raise NetworkError(message)

    # ------------------------------------------------------------ one attempt
    def _send(self, method: str, url: str, params: dict[str, Any] | None,
              body: Any) -> requests.Response:
        """Send once; return the response if its status is a success."""
        try:
            response = self.session.request(method, url, params=params, json=body,
                                            headers=self.headers, timeout=self.timeout_seconds)
        except requests.Timeout:
            raise _Retryable(f"timeout after {self.timeout_seconds}s") from None
        except requests.ConnectionError as error:
            raise _Retryable(f"connection error: {error}") from None

        status = response.status_code
        if status in RETRYABLE_STATUSES:
            raise _Retryable(f"status {status}", retry_after=_retry_after_seconds(response))
        if not 200 <= status < 300:
            body_text = response.text[:BODY_EXCERPT_CHARS]
            raise HttpStatusError(f"{method} {url}: status {status}: {body_text}", status, body_text)
        return response

    def _wait_seconds(self, attempt: int, problem: _Retryable) -> float:
        if problem.retry_after is not None:
            return problem.retry_after
        return self.backoff_seconds * 2 ** (attempt - 1)


def _content_problem(method: str, url: str, response: requests.Response, expected: str) -> str:
    content_type = response.headers.get("Content-Type", "")
    redirected = f", redirected to {response.url}" if response.history else ""
    return f"{method} {url}: expected {expected} but got '{content_type or 'no Content-Type'}'{redirected}"


def _parse_json(method: str, url: str, response: requests.Response) -> Any:
    if "json" not in response.headers.get("Content-Type", "").lower():
        raise InvalidResponseError(_content_problem(method, url, response, "JSON"))
    try:
        return response.json()
    except ValueError:
        raise _Retryable(f"unparseable JSON ({len(response.content)} bytes)", bad_body=True) from None


def _parse_text(method: str, url: str, response: requests.Response) -> str:
    # Servers label plain files differently (text/plain, application/octet-stream, ...);
    # the one answer that is certainly wrong is an HTML page (an error or login page).
    if "html" in response.headers.get("Content-Type", "").lower():
        raise InvalidResponseError(_content_problem(method, url, response, "a text file"))
    return response.text


def _retry_after_seconds(response: requests.Response) -> float | None:
    """The server's own 'try again in N seconds', if it sent a number."""
    value = response.headers.get("Retry-After", "")
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
