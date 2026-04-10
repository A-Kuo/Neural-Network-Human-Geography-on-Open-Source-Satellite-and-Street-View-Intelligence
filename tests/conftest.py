"""
conftest.py — Pytest configuration and fixtures.

Handles module-level mocking of packages that may not be installed (e.g., ratelimit).
"""

import sys
from unittest.mock import MagicMock

# Mock ratelimit module if not installed
if "ratelimit" not in sys.modules:
    ratelimit_mock = MagicMock()
    ratelimit_mock.limits = lambda calls, period: lambda f: f
    ratelimit_mock.sleep_and_retry = lambda f: f
    sys.modules["ratelimit"] = ratelimit_mock
