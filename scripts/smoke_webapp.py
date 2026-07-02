"""End-to-end smoke test for the public web app.

Runs against an already-launched uvicorn server (set DOCUMIND_BASE_URL, default
http://127.0.0.1:8000). Exits non-zero on the first failed assertion. Used by
the GitHub Actions ``smoke`` job, and by humans who just want a quick check
after `start-web.sh`.

We don't talk to Ollama here: ``/healthz`` may legitimately return 503 if the
Ollama server isn't reachable, and we tolerate that. We assert the JSON shape
and the public endpoints instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def main() -> int:
    base = os.environ.get("DOCUMIND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    failures: list[str] = []

    # /healthz: tolerate 200 (Ollama up) or 503 (degraded), but require the
    # JSON shape so the route is wired up correctly.
    code, body = _get(f"{base}/healthz")
    if code not in (200, 503):
        failures.append(f"/healthz: status {code}, expected 200/503")
    elif not isinstance(body, dict) or "ollama" not in body or "chroma" not in body:
        failures.append(f"/healthz: bad shape {body!r}")
    else:
        print(f"✓ /healthz -> {code} (ollama={body['ollama']}, chroma={body['chroma']})")

    # /api/sources: should always be a JSON object with a list (may be empty).
    code, body = _get(f"{base}/api/sources")
    if code != 200 or not isinstance(body, dict) or not isinstance(body.get("sources"), list):
        failures.append(f"/api/sources: bad response {code} {body!r}")
    else:
        print(f"✓ /api/sources -> {len(body['sources'])} document(s)")

    # /api/history: same idea.
    code, body = _get(f"{base}/api/history")
    if code != 200 or not isinstance(body, dict) or not isinstance(body.get("entries"), list):
        failures.append(f"/api/history: bad response {code} {body!r}")
    else:
        print(f"✓ /api/history -> {len(body['entries'])} recent entry/entries")

    # /chat without a question must 400.
    code, body = _post(f"{base}/chat", {"question": ""})
    if code != 400:
        failures.append(f"/chat empty: status {code}, expected 400")
    else:
        print(f"✓ /chat empty question -> 400")

    # /api/chat without a question must 400.
    code, body = _post(f"{base}/api/chat", {})
    if code != 400:
        failures.append(f"/api/chat empty: status {code}, expected 400")
    else:
        print(f"✓ /api/chat empty body -> 400")

    # /: SPA HTML.
    code, body = _get_raw(f"{base}/")
    if code != 200 or "<html" not in body.lower():
        failures.append(f"/: status {code}, expected 200 HTML")
    else:
        print(f"✓ / -> 200 HTML ({len(body)} bytes)")

    if failures:
        for f in failures:
            print(f"✗ {f}", file=sys.stderr)
        return 1
    print("All smoke checks passed.")
    return 0


def _post(url: str, data: dict, headers: dict | None = None) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def _get_raw(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Retry once after a short delay so the CI step doesn't race the server boot.
    rc = main()
    if rc != 0:
        time.sleep(2)
        rc = main()
    raise SystemExit(rc)
