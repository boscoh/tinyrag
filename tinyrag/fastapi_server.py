#!/usr/bin/env python3
"""FastAPI server for xConf Assistant's MCP client API."""

# IMPORTANT: Set up logging FIRST before any other imports that might trigger AWS calls
from tinyrag.setup_logger import setup_logging

setup_logging()

import asyncio
import logging
import os
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from tinyrag.config import chat_models, embed_models
from tinyrag.mcp_client import SpeakerMcpClient

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
mcp_client: Optional[SpeakerMcpClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global mcp_client

    llm_service = os.getenv("LLM_SERVICE")
    if not llm_service:
        logger.warning("LLM_SERVICE environment variable is not set, skipping MCP")
    else:
        try:
            mcp_client = SpeakerMcpClient(llm_service=llm_service)
            await mcp_client.connect()
            logger.info("MCP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")

    yield

    if mcp_client:
        try:
            await mcp_client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting MCP client: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TinyRAG",
        description="Tiny RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add routes
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        status = {
            "status": "ok",
        }
        if mcp_client and mcp_client.tools:
            status["mcp_tools"] = len(mcp_client.tools)
        return status

    @app.get("/info")
    async def get_info() -> Dict[str, Any]:
        info = {}
        if mcp_client and mcp_client.chat_client:
            info["llm_service"] = mcp_client.llm_service
            info["chat_model"] = getattr(
                mcp_client.chat_client, "model", None
            ) or chat_models.get(mcp_client.llm_service)
            info["embed_model"] = embed_models.get(mcp_client.llm_service)
        return info

    @app.get("/")
    async def simple():
        try:
            index_path = Path(__file__).parent / "index.html"
            return FileResponse(str(index_path), media_type="text/html")
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return {"error": "Could not load UI"}

    @app.post("/chat")
    @limiter.limit("10/minute")
    async def chat(request: Request, chat_request: ChatRequest) -> Dict[str, Any]:
        """
        Returns:
            {
                "id": <uuid4>,
                "role": "assistant",
                "status": "success",
                "data": <response from MCP client>
            }
        """
        try:
            history = (
                [msg.model_dump() for msg in chat_request.history]
                if chat_request.history
                else None
            )
            result = await mcp_client.process_query(chat_request.query, history=history)
            return {
                "id": str(uuid4()),
                "role": "assistant",
                "status": "success",
                "data": result,
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


class SlimMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    mode: str | None = None  # assistant or quest
    userToken: str | None = None
    history: list[SlimMessage] | None = None


app = create_app()


def is_in_container() -> bool:
    """Check if running inside a container (Docker, Podman, Kubernetes, ECS, etc.)."""
    if os.path.exists("/.dockerenv"):
        return True
    if os.path.exists("/run/.containerenv"):
        return True
    container_indicators = [
        "docker",
        "containerd",
        "kubepods",
        "crio",
        "libpod",
        "ecs",
    ]
    for cgroup_file in ["/proc/1/cgroup", "/proc/self/cgroup"]:
        if os.path.exists(cgroup_file):
            try:
                with open(cgroup_file, "r") as f:
                    content = f.read()
                    if any(indicator in content for indicator in container_indicators):
                        return True
            except (OSError, IOError):
                pass
    return False


def poll_and_open_browser(
    port: int, timeout_seconds: int = 300, interval_seconds: int = 1
) -> None:
    start_time = time.time()
    ui_url = f"http://localhost:{port}"

    while time.time() - start_time < timeout_seconds:
        try:
            response = httpx.get(ui_url, timeout=2)
            if response.status_code == 200 and "<html" in response.text.lower():
                logger.info(f"Server is live at {ui_url}, opening browser...")
                webbrowser.open(ui_url)
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        time.sleep(interval_seconds)

    logger.warning(f"Server did not respond within {timeout_seconds} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument(
        "--port", type=int, default=80, help="Port to run the server on"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    logger.info(f"Args: {args}")

    if not is_in_container():
        poller_thread = threading.Thread(
            target=poll_and_open_browser, args=(args.port,), daemon=True
        )
        poller_thread.start()
    else:
        logger.info("Running in container, skipping browser auto-open")

    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
        # Use existing logging setup from setup_logger
        access_log=False,  # Disable uvicorn access logs to avoid duplication
    )
