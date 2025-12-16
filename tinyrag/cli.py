#!/usr/bin/env python3
"""
TinyRAG CLI - Command-line interface for TinyRAG
"""

import asyncio
import os
import threading
import time
import webbrowser

import httpx
import typer
import uvicorn

from tinyrag.fastapi_server import create_app
from tinyrag.mcp_client import amain
from tinyrag.run_docker import main as run_docker_main

app = typer.Typer(
    name="tinyrag", help="TinyRAG - Tiny RAG starter kit", no_args_is_help=True
)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
):
    """Start the UI with FastAPI backend"""
    fastapi_app = create_app()

    if not no_browser:

        def wait_and_open_browser():
            base_url = f"http://{host}:{port}"
            max_retries = 60
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = httpx.get(f"{base_url}/ready", timeout=1)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("ready"):
                            webbrowser.open(base_url)
                            typer.echo(f"Opening {base_url} in browser...")
                            return
                except Exception:
                    pass
                
                time.sleep(0.5)
                retry_count += 1
            
            # Timeout reached, open browser anyway
            try:
                webbrowser.open(base_url)
                typer.echo(f"Opening {base_url} in browser (timeout waiting for ready)...")
            except Exception as e:
                typer.echo(f"Could not open browser: {e}")

        thread = threading.Thread(target=wait_and_open_browser, daemon=True)
        thread.start()

    typer.echo(f"Starting TinyRAG UI on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start the FastAPI server only"""
    fastapi_app = create_app()
    typer.echo(f"Starting FastAPI server on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def mcp():
    """Start MCP client"""
    service = os.getenv("CHAT_SERVICE")
    if not service:
        typer.echo("Error: CHAT_SERVICE environment variable is not set")
        raise typer.Exit(1)
    asyncio.run(amain(service))


@app.command()
def docker():
    """Build and run Docker container with AWS credentials"""
    run_docker_main()


@app.command()
def version():
    """Show version"""
    typer.echo("tinyrag 0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
