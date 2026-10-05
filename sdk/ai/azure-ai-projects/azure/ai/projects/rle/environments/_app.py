# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
"""OpenEnv app factory with request-scoped Foundry session routing."""

import json
import re
from typing import Any, Callable, Optional, Type
from urllib.parse import unquote_to_bytes

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app as _openenv_create_app  # type: ignore[import-untyped]
from openenv.core.env_server.interfaces import Environment  # type: ignore[import-untyped]
from openenv.core.env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse  # type: ignore[import-untyped]
from openenv.core.env_server.types import (  # type: ignore[import-untyped]
    Action,
    ConcurrencyConfig,
    Observation,
    State,
)
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# OpenEnv 0.6 has no HTTP body limit. Match its client's default message bound.
_MAX_BODY_BYTES = 100 * 1024 * 1024
_INVALID_ESCAPE = re.compile(rb"%(?![0-9a-fA-F]{2})")


class _SessionQueryMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path) :]
        elif root_path and path == root_path:
            path = ""
        if path != "/mcp":
            await self.app(scope, receive, send)
            return

        remaining = []
        values = []
        for part in scope.get("query_string", b"").split(b"&"):
            key, _, value = part.partition(b"=")
            if unquote_to_bytes(key.replace(b"+", b" ")) == b"rle_session_id":
                values.append(value)
            else:
                remaining.append(part)
        if not values:
            await self.app(scope, receive, send)
            return

        async def error(code: JsonRpcErrorCode, message: str, status_code: int = 200) -> None:
            response = JSONResponse(
                JsonRpcResponse.error_response(code, message).model_dump(),
                status_code=status_code,
            )
            await response(scope, receive, send)

        if len(values) != 1 or not values[0]:
            await error(JsonRpcErrorCode.INVALID_PARAMS, "Exactly one nonempty rle_session_id is required")
            return
        try:
            if _INVALID_ESCAPE.search(values[0]):
                raise ValueError("Invalid query escaping")
            session_id = unquote_to_bytes(values[0].replace(b"+", b" ")).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            await error(JsonRpcErrorCode.INVALID_PARAMS, "Invalid rle_session_id encoding")
            return

        headers = scope.get("headers", [])
        for key, value in headers:
            if key.lower() == b"content-encoding" and value.strip().lower() != b"identity":
                await error(JsonRpcErrorCode.INVALID_REQUEST, "Encoded MCP request bodies are not supported")
                return
            if key.lower() == b"content-length":
                try:
                    length = int(value)
                except ValueError:
                    await error(JsonRpcErrorCode.INVALID_REQUEST, "Invalid Content-Length")
                    return
                if length < 0:
                    await error(JsonRpcErrorCode.INVALID_REQUEST, "Invalid Content-Length")
                    return
                if length > _MAX_BODY_BYTES:
                    await error(JsonRpcErrorCode.INVALID_REQUEST, "MCP request body is too large", 413)
                    return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > _MAX_BODY_BYTES:
                await error(JsonRpcErrorCode.INVALID_REQUEST, "MCP request body is too large", 413)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        adapted_scope = dict(scope, query_string=b"&".join(remaining))
        try:
            request = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            request = None
        # Invalid JSON, batches, and invalid params retain OpenEnv's own errors.
        if isinstance(request, dict) and request.get("method") in ("tools/list", "tools/call"):
            params = request.get("params", {})
            if isinstance(params, dict):
                params["session_id"] = session_id
                request["params"] = params
                body = bytearray(json.dumps(request, ensure_ascii=True).encode("utf-8"))
                adapted_scope["headers"] = [
                    (key, value)
                    for key, value in headers
                    if key.lower() not in (b"content-length", b"transfer-encoding")
                ] + [(b"content-length", str(len(body)).encode("ascii"))]

        delivered = False

        async def adapted_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(adapted_scope, adapted_receive, send)


def create_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
    custom_tab_name: str = "Custom",
    custom_tab_primary: bool = False,
    show_default_tab: bool = True,
    title_override: Optional[str] = None,
    state_cls: Type[State] = State,
) -> FastAPI:
    """Create an OpenEnv app with Foundry MCP session routing installed.

    Accepts the same arguments and returns the same FastAPI app as OpenEnv 0.6's
    ``create_app``. A single nonempty ``rle_session_id`` query value on HTTP
    ``POST /mcp`` overrides ``params.session_id`` for ``tools/list`` and
    ``tools/call`` only. Requests without that query value remain unchanged.
    Query-adapted bodies are limited to 100 MiB; compressed bodies are rejected.
    The query selects a session, not authorization or session ownership.

    :param env: Factory that creates environment instances.
    :param action_cls: Action model accepted by the environment.
    :param observation_cls: Observation model returned by the environment.
    :param env_name: Optional environment name.
    :param max_concurrent_envs: Optional maximum concurrent session count.
    :param concurrency_config: Optional advanced concurrency configuration.
    :param gradio_builder: Optional custom Gradio UI builder.
    :param custom_tab_name: Custom UI tab label.
    :param custom_tab_primary: Whether the custom tab is initially active.
    :param show_default_tab: Whether to show the default UI tab.
    :param title_override: Optional UI title.
    :param state_cls: State model reported by the environment.
    :return: OpenEnv FastAPI application with session query middleware.
    :rtype: ~fastapi.FastAPI
    """
    app = _openenv_create_app(
        env,
        action_cls,
        observation_cls,
        env_name,
        max_concurrent_envs,
        concurrency_config,
        gradio_builder,
        custom_tab_name,
        custom_tab_primary,
        show_default_tab,
        title_override,
        state_cls,
    )
    app.add_middleware(_SessionQueryMiddleware)
    return app
