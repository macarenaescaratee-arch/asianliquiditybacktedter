"""
Minimal HTTP stub: OANDA v20 ``GET /v3/instruments/EUR_USD/candles`` only.

Advances a synthetic hour index on each request so polling sees new closed bars.
For local verification when ``OANDA_REST_BASE`` points here.

Usage::

    python scripts/mock_oanda_candles_server.py 18999
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

ANCHOR = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
_latest_hour_idx = 0
_request_count = 0


def _candles(count: int) -> list[dict]:
    global _latest_hour_idx
    _latest_hour_idx += 1
    out: list[dict] = []
    base_px = 1.0820
    for j in range(count):
        h = _latest_hour_idx - (count - 1 - j)
        ts = ANCHOR + timedelta(hours=h)
        w = 0.0045 * math.sin(h / 18.0)
        o = base_px + w
        c = o + 0.00025 * math.sin(h / 7.0)
        hi = max(o, c) + 0.00045
        lo = min(o, c) - 0.00045
        out.append(
            {
                "complete": True,
                "time": ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "mid": {
                    "o": f"{o:.5f}",
                    "h": f"{hi:.5f}",
                    "l": f"{lo:.5f}",
                    "c": f"{c:.5f}",
                },
                "volume": 1000,
            }
        )
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def do_GET(self) -> None:
        global _request_count
        _request_count += 1
        u = urlparse(self.path)
        if u.path == "/v3/instruments/EUR_USD/candles":
            fail_first = int(os.environ.get("MOCK_OANDA_FAIL_FIRST", "0"))
            if _request_count <= fail_first:
                raw = json.dumps({"errorMessage": "synthetic temporary failure"}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            qs = parse_qs(u.query)
            count = int(qs.get("count", ["500"])[0])
            count = max(1, min(count, 5000))
            body = {"candles": _candles(count)}
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(404)
        self.end_headers()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18999
    server = HTTPServer(("127.0.0.1", port), _Handler)
    print(f"mock OANDA candles on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
