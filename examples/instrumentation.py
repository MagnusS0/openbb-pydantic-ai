"""
This example shows how to set up Logfire instrumentation for Pydantic AI.
By default traces are sent to Logfire. To send to any OTEL-compatible
backend (e.g. Aspire, Jaeger) instead, set `OTEL_EXPORTER_OTLP_ENDPOINT`
and call with `send_to_logfire=False`.
"""

from __future__ import annotations

import os

import logfire


def setup_observability(*, send_to_logfire: bool = True) -> None:
    """
    Configure Logfire instrumentation for Pydantic AI.

    Set `send_to_logfire=False` to route traces to a custom OTEL endpoint
    configured via the `OTEL_EXPORTER_OTLP_ENDPOINT` env var.
    """
    if not send_to_logfire:
        # Ensure the env var is set so logfire knows where to export
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            msg = "OTEL_EXPORTER_OTLP_ENDPOINT must be set when send_to_logfire=False"
            raise RuntimeError(msg)

    logfire.configure(send_to_logfire=send_to_logfire)
    logfire.instrument_pydantic_ai()
