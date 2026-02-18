"""Minimal example: connect a Pydantic AI agent to OpenBB Workspace."""

from __future__ import annotations

import os
from pathlib import Path

from anyio import BrokenResourceError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from pydantic_ai import Agent
from pydantic_ai_skills import SkillsToolset

# Optional: uncomment to enable observability (requires `observability` extra)
# from examples.instrumentation import setup_observability
from examples.prompts.skills_example_prompt import OPENBB_WORKSPACE_SYSTEM_PROMPT
from openbb_pydantic_ai import OpenBBAIAdapter, OpenBBDeps

# setup_observability()

# Pydantic AI model string: "provider:model_name"
AGENT_MODEL = os.getenv("AGENT_MODEL", "openrouter:moonshotai/kimi-k2.5")

_EXAMPLE_DIR = Path(__file__).resolve().parent
skills_toolset = SkillsToolset(directories=[_EXAMPLE_DIR / "skills"])

agent = Agent(
    AGENT_MODEL,
    instructions=OPENBB_WORKSPACE_SYSTEM_PROMPT,
    deps_type=OpenBBDeps,
    toolsets=[skills_toolset],
    retries=3,
)

app = FastAPI(default_response_class=ORJSONResponse)

AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8003")


@app.get("/agents.json")
async def agents_json() -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "example-agent": {
                "name": "Example Agent with Skills",
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
