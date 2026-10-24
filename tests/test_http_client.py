"""Tests for the HTTP layer. No network: a fake session plays a scripted list of answers."""
import json
import logging
import unittest

import requests

from data_factory.exceptions import HttpStatusError, InvalidResponseError, NetworkError
from data_factory.http_client import HttpClient

URL = "https://example.test/resource.json"


class FakeResponse:
    def __init__(self, status=200, body="[]", content_type="application/json",
                 headers=None, redirected_to=None):
        self.status_code = status
        self.text = body
        self.content = body.encode("utf-8")
        self.headers = {"Content-Type": content_type, **(headers or {})}
        self.history = [object()] if redirected_to else []
        self.url = redirected_to or URL

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Answers requests with the next scripted item: a FakeResponse or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.requests.append({"method": method, "url": url, "params": params, "json": json})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*script, max_retries=3):
    waits = []
    session = FakeSession(*script)
    http = HttpClient(timeout_seconds=5, max_retries=max_retries, backoff_seconds=2,
                      session=session, sleep=waits.append)
    return http, session, waits


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    # -------------------------------------------------------------- success
    def test_get_returns_parsed_json(self):
        http, session, waits = client(FakeResponse(body='[{"a": "1"}]'))
        self.assertEqual(http.get_json(URL, {"$limit": 1}), [{"a": "1"}])
        self.assertEqual(session.requests[0]["method"], "GET")
        self.assertEqual(session.requests[0]["params"], {"$limit": 1})
        self.assertEqual((http.stats.calls, http.stats.attempts, waits), (1, 1, []))

    def test_post_sends_the_body_as_json(self):
        http, session, _ = client(FakeResponse(body='{"rows": []}'))
        result = http.post_json(URL, {"query": "SELECT *"})
        self.assertEqual(result, {"rows": []})
        self.assertEqual(session.requests[0]["method"], "POST")
        self.assertEqual(session.requests[0]["json"], {"query": "SELECT *"})

    def test_content_type_with_charset_is_accepted(self):
        http, _, _ = client(FakeResponse(content_type="application/json; charset=utf-8"))
        self.assertEqual(http.get_json(URL), [])

    # -------------------------------------------------------------- retries
    def test_server_error_is_retried_with_exponential_backoff(self):
        http, _, waits = client(FakeResponse(503), FakeResponse(500), FakeResponse(body="[1]"))
        self.assertEqual(http.get_json(URL), [1])
        self.assertEqual(waits, [2, 4])
        self.assertEqual((http.stats.attempts, http.stats.retries), (3, 2))

    def test_rate_limit_respects_retry_after(self):
        http, _, waits = client(FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse())
        http.get_json(URL)
        self.assertEqual(waits, [7.0])

    def test_timeouts_and_connection_errors_are_retried(self):
        http, _, _ = client(requests.Timeout(), requests.ConnectionError("reset"), FakeResponse())
        self.assertEqual(http.get_json(URL), [])
        self.assertEqual(http.stats.attempts, 3)

    def test_gives_up_after_max_retries_with_network_error(self):
        http, session, waits = client(*[requests.Timeout()] * 4, max_retries=3)
        with self.assertRaises(NetworkError) as caught:
            http.get_json(URL)
        self.assertNotIsInstance(caught.exception, InvalidResponseError)
        self.assertEqual(len(session.requests), 4)       # 1 attempt + 3 retries
        self.assertEqual(waits, [2, 4, 8])
        self.assertIn("after 4 attempts", str(caught.exception))

    def test_truncated_json_is_retried_then_reported_as_invalid(self):
        broken = '[{"unique_key": "1"'
        http, _, _ = client(*[FakeResponse(body=broken)] * 3, max_retries=2)
        with self.assertRaises(InvalidResponseError):
            http.get_json(URL)
        self.assertEqual(http.stats.attempts, 3)

    def test_truncated_json_recovers_if_the_next_answer_is_complete(self):
        http, _, _ = client(FakeResponse(body='[{"a"'), FakeResponse(body='[{"a": 1}]'))
        self.assertEqual(http.get_json(URL), [{"a": 1}])

    # -------------------------------------------------------------- no retry
    def test_client_error_status_is_not_retried(self):
        http, session, waits = client(FakeResponse(400, body='{"message": "No such column: zip"}'))
        with self.assertRaises(HttpStatusError) as caught:
            http.get_json(URL)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("No such column", caught.exception.body)
        self.assertEqual((len(session.requests), waits), (1, []))

    def test_html_with_status_200_is_invalid_and_not_retried(self):
        # The Census API case: no key -> redirect to an HTML page answered with status 200.
        page = FakeResponse(200, body="<html>missing key</html>", content_type="text/html",
                            redirected_to="https://example.test/missing_key.html")
        http, session, _ = client(page)
        with self.assertRaises(InvalidResponseError) as caught:
            http.get_json(URL)
        self.assertIn("missing_key.html", str(caught.exception))
        self.assertEqual(len(session.requests), 1)

    # -------------------------------------------------------------- text files
    def test_text_download_returns_the_body(self):
        http, _, _ = client(FakeResponse(body="A|B\n1|2\n", content_type="text/plain"))
        self.assertEqual(http.get_text(URL), "A|B\n1|2\n")

    def test_text_download_refuses_an_html_page(self):
        http, _, _ = client(FakeResponse(body="<html>error</html>", content_type="text/html"))
        with self.assertRaises(InvalidResponseError):
            http.get_text(URL)

    def test_text_download_is_retried_like_any_request(self):
        http, _, waits = client(FakeResponse(503), FakeResponse(body="x", content_type="text/plain"))
        self.assertEqual(http.get_text(URL), "x")
        self.assertEqual(waits, [2])

    # -------------------------------------------------------------- counting
    def test_stats_accumulate_across_calls(self):
        http, _, _ = client(FakeResponse(), FakeResponse(503), FakeResponse())
        http.get_json(URL)
        http.get_json(URL)
        self.assertEqual((http.stats.calls, http.stats.attempts, http.stats.retries), (2, 3, 1))


if __name__ == "__main__":
    unittest.main()
