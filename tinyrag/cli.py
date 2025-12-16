#!/usr/bin/env python3
"""
TinyRAG CLI - Command-line interface for TinyRAG
"""

import time
import webbrowser

import typer

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
    import uvicorn

    from tinyrag.fastapi_server import create_app

    # Create app instance
    fastapi_app = create_app()

    # Open browser after a brief delay to allow server startup
    if not no_browser:

        def open_browser():
            time.sleep(2)
            try:
                url = f"http://{host}:{port}"
                webbrowser.open(url)
                typer.echo(f"Opening {url} in browser...")
            except Exception as e:
                typer.echo(f"Could not open browser: {e}")

        import threading

        thread = threading.Thread(target=open_browser, daemon=True)
        thread.start()

    typer.echo(f"Starting TinyRAG UI on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start the FastAPI server only"""
    import uvicorn

    from tinyrag.fastapi_server import create_app

    fastapi_app = create_app()
    typer.echo(f"Starting FastAPI server on http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def mcp():
    """Start MCP client"""
    import asyncio
    import os

    from tinyrag.mcp_client import amain

    service = os.getenv("LLM_SERVICE", "openai")
    asyncio.run(amain(service))


@app.command()
def docker():
    """Build and run Docker container with AWS credentials"""
    from tinyrag.run_docker import main

    main()


@app.command()
def version():
    """Show version"""
    typer.echo("tinyrag 0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
