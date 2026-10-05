# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

import asyncio
import inspect
import json
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

pytest.importorskip("openenv")
pytest.importorskip("fastmcp")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server import http_server
from openenv.core.env_server.types import Observation, State

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment, create_app
from azure.ai.projects.rle.environments import _app


class _SessionEnvironment(RLEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.closes = 0

        @self.tool()
        def count(value: str, session_id: str = "tool-argument"):
            self.calls += 1
            return {
                "episode": self.state.episode_id,
                "calls": self.calls,
                "value": value,
                "session_id": session_id,
            }

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._set_state(State(episode_id=episode_id))
        return Observation()

    def grade(self, action, timeout_s=None, **kwargs):
        return Observation(done=True, reward=float(self.calls), metadata={"episode": self.state.episode_id})

    def close(self):
        self.closes += 1
        super().close()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    created = []

    def factory():
        environment = _SessionEnvironment()
        created.append(environment)
        return environment

    app = create_app(factory, GradeAction, Observation, max_concurrent_envs=2)
    app.state.created = created
    with TestClient(app) as test_client:
        yield test_client


def _rpc(client, method, params=None, query="", request_id="request"):
    request = {"jsonrpc": "2.0", "method": method, "id": request_id}
    if params is not None:
        request["params"] = params
    return client.post("/mcp" + query, json=request).json()


def _create_session(client):
    return _rpc(client, "openenv/session/create")["result"]["session_id"]


def _reset(client, session_id, episode):
    with client.websocket_connect("/ws?" + urlencode({"session_id": session_id})) as socket:
        socket.send_json({"type": "reset", "data": {"episode_id": episode}})
        response = socket.receive_json()
        assert response["type"] == "observation", response


def test_real_openenv_two_sessions_overwrite_and_repeated_calls(client):
    first, second = _create_session(client), _create_session(client)
    _reset(client, first, "first-episode")
    _reset(client, second, "second-episode")
    query = "?" + urlencode({"rle_session_id": first, "other": "a+b"})

    listed = _rpc(client, "tools/list", {"session_id": second}, query, 17)
    assert listed["id"] == 17
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["count"]
    schema = listed["result"]["tools"][0]["inputSchema"]
    assert set(schema["properties"]) == {"value", "session_id"}
    assert schema["properties"]["session_id"]["default"] == "tool-argument"
    assert "session_id" not in schema.get("required", [])

    for call in (1, 2):
        response = _rpc(
            client,
            "tools/call",
            {"session_id": second, "name": "count", "arguments": {"value": "hello", "session_id": "argument"}},
            query,
        )
        assert response["result"]["data"] == {
            "episode": "first-episode",
            "calls": call,
            "value": "hello",
            "session_id": "argument",
        }
    assert (
        _rpc(client, "tools/call", {"session_id": second, "name": "count", "arguments": {"value": "body"}})["result"][
            "data"
        ]["episode"]
        == "second-episode"
    )

    with client.websocket_connect("/ws?" + urlencode({"session_id": first})) as socket:
        socket.send_json({"type": "step", "data": {"answer": "answer"}})
        graded = socket.receive_json()["data"]
        assert graded["done"] is True
        assert graded["reward"] == 2.0
        assert graded["observation"]["metadata"]["episode"] == "first-episode"
    environment = next(env for env in client.app.state.created if env.state.episode_id == "first-episode")
    assert environment.closes == 0
    assert _rpc(client, "openenv/session/close", {"session_id": first})["result"]["closed"] is True
    assert environment.closes == 1
    assert _rpc(client, "openenv/session/close", {"session_id": second})["result"]["closed"] is True


def test_real_openenv_url_decoding_once_and_missing_params(client, monkeypatch):
    session_id = "session /+?&=%2Fé"
    with monkeypatch.context() as patch:
        patch.setattr(http_server, "uuid", SimpleNamespace(uuid4=lambda: session_id))
        assert _create_session(client) == session_id
    _reset(client, session_id, "encoded-episode")
    query = "?" + urlencode({"rle_session_id": session_id})
    assert _rpc(client, "tools/list", query=query)["result"]["tools"][0]["name"] == "count"
    assert (
        _rpc(client, "tools/call", {"name": "count", "arguments": {"value": "encoded"}}, query)["result"]["data"][
            "episode"
        ]
        == "encoded-episode"
    )
    assert _rpc(client, "openenv/session/close", {"session_id": session_id})["result"]["closed"]


@pytest.mark.parametrize(
    "query",
    [
        "?rle_session_id=",
        "?rle_session_id",
        "?rle_session_id=a&rle_session_id=b",
        "?rle_session_id=a&%72le_session_id=a",
        "?rle_session_id=%FF",
        "?rle_session_id=%2",
    ],
)
def test_invalid_session_query_returns_runtime_error(client, query):
    response = _rpc(client, "tools/list", query=query)
    assert response["error"]["code"] == -32602
    assert response["id"] is None


@pytest.mark.parametrize("params", [None, [], "bad", 1, True])
def test_invalid_params_not_replaced(client, params):
    response = client.post(
        "/mcp?rle_session_id=valid", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": params}
    ).json()
    assert response["error"]["code"] == -32600


@pytest.mark.parametrize("body", [b"{", b"null", b"[]", b'[{"jsonrpc":"2.0","method":"tools/list","id":1}]', b"\xff"])
def test_malformed_and_batch_requests_retain_upstream_error(client, body):
    plain = client.post("/mcp", content=body).json()
    adapted = client.post("/mcp?rle_session_id=valid", content=body).json()
    assert adapted == plain
    assert adapted["error"]["code"] == -32700


@pytest.mark.parametrize("method", ["initialize", "notifications/initialized"])
def test_other_methods_retain_upstream_behavior(client, method):
    assert _rpc(client, method, query="?rle_session_id=not-a-session") == _rpc(client, method)
    assert _rpc(client, method)["error"]["code"] == -32601


def test_tool_notification_retains_upstream_response(client):
    session = _create_session(client)
    response = client.post(
        "/mcp?" + urlencode({"rle_session_id": session}), json={"jsonrpc": "2.0", "method": "tools/list"}
    ).json()
    assert response["id"] is None
    assert response["result"]["tools"][0]["name"] == "count"
    _rpc(client, "openenv/session/close", {"session_id": session})


def test_lifecycle_not_given_query_session(client):
    session = _create_session(client)
    assert (
        _rpc(client, "openenv/session/close", query="?" + urlencode({"rle_session_id": session}))["error"]["code"]
        == -32602
    )
    assert _rpc(client, "tools/list", {"session_id": session})["result"]["tools"]
    _rpc(client, "openenv/session/close", {"session_id": session})


def test_non_mcp_and_websocket_remain_unchanged(client):
    assert client.get("/health?rle_session_id=").status_code == 200
    with client.websocket_connect("/mcp?rle_session_id=") as socket:
        socket.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": 5})
        assert socket.receive_json()["result"]["tools"][0]["name"] == "count"


def test_factory_contract_matches_and_forwards_openenv(monkeypatch):
    expected = inspect.signature(http_server.create_app)
    actual = inspect.signature(create_app)
    assert [(p.name, p.kind, p.default) for p in actual.parameters.values()] == [
        (p.name, p.kind, p.default) for p in expected.parameters.values()
    ]
    forwarded = []
    app = FastAPI()

    def upstream(*args):
        forwarded.extend(args)
        return app

    monkeypatch.setattr(_app, "_openenv_create_app", upstream)
    values = [
        _SessionEnvironment,
        GradeAction,
        Observation,
        "test",
        2,
        None,
        None,
        "custom",
        True,
        False,
        "title",
        State,
    ]
    assert create_app(*values) is app
    assert forwarded == values
    assert [middleware.cls for middleware in app.user_middleware] == [_app._SessionQueryMiddleware]


async def _invoke(body, query=b"rle_session_id=query", scope_updates=None, headers=None, messages=None):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "root_path": "",
        "query_string": query,
        "headers": headers or [],
    }
    scope.update(scope_updates or {})
    incoming = list(
        messages
        if messages is not None
        else [
            {"type": "http.request", "body": body[:5], "more_body": True},
            {"type": "http.request", "body": body[5:], "more_body": False},
            {"type": "http.disconnect"},
        ]
    )
    sent = []
    received = []

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    async def downstream(adapted_scope, adapted_receive, adapted_send):
        received.append(adapted_scope)
        received.append(await adapted_receive())
        await adapted_send({"type": "http.response.start", "status": 200, "headers": []})
        await adapted_send({"type": "http.response.body", "body": b"first", "more_body": True})
        await adapted_send({"type": "http.response.body", "body": b"second", "more_body": False})
        received.append(await adapted_receive())

    await _app._SessionQueryMiddleware(downstream)(scope, receive, send)
    return scope, received, sent


@pytest.mark.asyncio
async def test_asgi_query_consumed_headers_fixed_and_response_streaming():
    body = b'{"jsonrpc":"2.0","method":"tools/list","id":42}'
    original, received, sent = await _invoke(
        body,
        query=b"other=a%2Bb&%72le_session_id=query%2B%252F&flag&other=two",
        headers=[(b"content-length", str(len(body)).encode()), (b"x-other", b"keep")],
    )
    assert received[0]["query_string"] == b"other=a%2Bb&flag&other=two"
    adapted = received[1]["body"]
    assert json.loads(adapted)["params"] == {"session_id": "query+%2F"}
    assert dict(received[0]["headers"]) == {b"content-length": str(len(adapted)).encode(), b"x-other": b"keep"}
    assert original["query_string"].startswith(b"other=a%2Bb&%72le_session_id")
    assert received[2] == {"type": "http.disconnect"}
    assert sent[1:] == [
        {"type": "http.response.body", "body": b"first", "more_body": True},
        {"type": "http.response.body", "body": b"second", "more_body": False},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates,query",
    [
        ({}, b"other=unchanged"),
        ({"type": "websocket"}, b"rle_session_id="),
        ({"path": "/health"}, b"rle_session_id="),
        ({"method": "GET"}, b"rle_session_id="),
    ],
)
async def test_asgi_bypass_preserves_scope_and_receive(updates, query):
    scope, received, _ = await _invoke(b"original bytes", query=query, scope_updates=updates)
    assert received[0] is scope
    assert received[1] == {"type": "http.request", "body": b"origi", "more_body": True}
    assert received[2]["body"] == b"nal bytes"


@pytest.mark.asyncio
async def test_asgi_mounted_route():
    _, received, _ = await _invoke(b'{"method":"tools/list"}', scope_updates={"root_path": "/env", "path": "/env/mcp"})
    assert json.loads(received[1]["body"])["params"]["session_id"] == "query"


@pytest.mark.asyncio
async def test_concurrent_queries_do_not_share_session_state():
    first, second = await asyncio.gather(
        _invoke(b'{"method":"tools/list"}', query=b"rle_session_id=first"),
        _invoke(b'{"method":"tools/list"}', query=b"rle_session_id=second"),
    )
    assert json.loads(first[1][1]["body"])["params"]["session_id"] == "first"
    assert json.loads(second[1][1]["body"])["params"]["session_id"] == "second"


@pytest.mark.asyncio
async def test_root_path_prefix_is_not_a_mount_boundary():
    _, received, _ = await _invoke(b'{"method":"tools/list"}', scope_updates={"root_path": "/m", "path": "/mcp"})
    assert json.loads(received[1]["body"])["params"]["session_id"] == "query"


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [[], [(b"content-length", b"500")]])
async def test_bounded_body_with_and_without_content_length(monkeypatch, headers):
    monkeypatch.setattr(_app, "_MAX_BODY_BYTES", 10)
    _, received, sent = await _invoke(b"body exceeding bound", headers=headers)
    assert received == []
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == -32600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [[(b"content-encoding", b"gzip")], [(b"content-length", b"bad")], [(b"content-length", b"-1")]]
)
async def test_invalid_body_encoding_and_length(headers):
    _, received, sent = await _invoke(b"body", headers=headers)
    assert received == []
    assert json.loads(sent[1]["body"])["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_disconnect_during_body_not_dispatched():
    _, received, sent = await _invoke(
        b"",
        messages=[
            {"type": "http.request", "body": b"{", "more_body": True},
            {"type": "http.disconnect"},
        ],
    )
    assert received == []
    assert sent == []


@pytest.mark.asyncio
async def test_cancellation_propagates():
    async def cancelled():
        raise asyncio.CancelledError()

    async def unexpected(*args):
        pytest.fail("Disconnected request was dispatched or answered")

    with pytest.raises(asyncio.CancelledError):
        await _app._SessionQueryMiddleware(unexpected)(
            {"type": "http", "method": "POST", "path": "/mcp", "query_string": b"rle_session_id=query"},
            cancelled,
            unexpected,
        )
