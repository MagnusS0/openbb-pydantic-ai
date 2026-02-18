"""Minimal example: connect a Pydantic AI agent to OpenBB Workspace."""

from __future__ import annotations

import os
from textwrap import dedent

from anyio import BrokenResourceError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from pydantic_ai import Agent

# Optional: uncomment to enable observability (requires `observability` extra)
# from examples.instrumentation import setup_observability
from openbb_pydantic_ai import OpenBBAIAdapter, OpenBBDeps

# setup_observability()

# Pydantic AI model string: "provider:model_name"
AGENT_MODEL = os.getenv("AGENT_MODEL", "openrouter:moonshotai/kimi-k2.5")

agent = Agent(
    AGENT_MODEL,
    instructions=dedent("""
        You are a helpful data analyst and financial assistant connected to
        OpenBB Workspace. Use the available widget tools to fetch data, build
        charts, and answer the user's questions.

        You will have a set of tools available that allow you to interact
        with OpenBB Workspace. Use these tools to fetch data, create charts, and perform
        other actions as needed to answer the user's questions.

        When you create a chart artifact, you can place it in the output by using
        a {{place_chart_here}} placholder in your final response.
        It will be rendered as an actual chart in the OpenBB Workspace interface.

        Never make up information, if unsure, use the tools to find out.
    """).strip(),
    deps_type=OpenBBDeps,
    retries=3,
)

app = FastAPI(default_response_class=ORJSONResponse)

AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8003")


@app.get("/agents.json")
async def agents_json() -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "example-agent": {
                "name": "Example Agent",
                "description": (
                    "A data analyst and financial assistant for OpenBB Workspace."
                ),
                "endpoints": {
                    "query": f"{AGENT_BASE_URL}/query",
                },
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": True,
                    "mcp-tools": True,
                },
            }
        }
    )


@app.post("/query")
async def query(request: Request):
    try:
        return await OpenBBAIAdapter.dispatch_request(request, agent=agent)
    except BrokenResourceError:
        # Client disconnected — normal for SSE streams
        pass


app.add_middleware(GZipMiddleware, minimum_size=1000)  # ty:ignore[invalid-argument-type]
app.add_middleware(
    CORSMiddleware,  # ty:ignore[invalid-argument-type]
    allow_origins=[
        "https://pro.openbb.co",
        "http://localhost:1420",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
