"""Tests for the RLE Execute Rollout operation.

These run the real client pipeline against a local HTTP server rather than a mocked transport,
so the URL, query string, headers and body under assertion are the bytes that would go on the
wire -- the things that a stub of our own construction would simply agree with us about.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import pytest

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import RLERolloutModelBinding
from azure.ai.projects.operations import RLERolloutError
from azure.ai.projects.operations._patch_rle_rollout import (
    EXECUTE_ROLLOUT_API_VERSION,
    FORWARDED_TOKEN_HEADER,
    build_rollout_body,
)
from azure.core.credentials import AccessToken

_TOKEN = "test-forwarded-token"


class _FakeCredential:
    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        self.scopes = scopes
        return AccessToken(_TOKEN, 9999999999)

    def close(self) -> None:
        pass


class _Recorder:
    """One captured request plus the canned reply to answer it with."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.status = 200
        self.body: Any = {"rollout_id": "unset"}


def _serve(recorder: _Recorder) -> "tuple[HTTPServer, str]":
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's required spelling
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            recorder.requests.append(
                {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(raw) if raw else None,
                }
            )
            payload = json.dumps(recorder.body).encode()
            self.send_response(recorder.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[0], server.server_address[1]
    return server, f"http://{host}:{port}/api/projects/proj"


@pytest.fixture
def rollout_server():
    recorder = _Recorder()
    server, endpoint = _serve(recorder)
    try:
        yield recorder, endpoint
    finally:
        server.shutdown()
        server.server_close()


def _client(endpoint: str) -> AIProjectClient:
    return AIProjectClient(
        endpoint=endpoint,
        credential=_FakeCredential(),
        allow_preview=True,
    )


def _rollout(client: AIProjectClient, *args: Any, **kwargs: Any):
    """Call the operation against the loopback server.

    ``enforce_https=False`` is azure-core's per-request escape from the bearer policy's TLS
    guard, needed only because this server is plain HTTP. It travels through the operation's
    ``**kwargs``, so these tests also pin that pipeline options reach the transport.
    """
    return client.rle.execute_rollout(*args, enforce_https=False, **kwargs)


def _binding(**overrides: Any) -> RLERolloutModelBinding:
    values: Dict[str, Any] = {
        "model_name": "Qwen/Qwen3-32B",
        "project_endpoint": "https://acct.services.ai.azure.com/api/projects/proj",
        "loom_session_id": "session-1",
        "checkpoint_id": "checkpoint-1",
    }
    values.update(overrides)
    return RLERolloutModelBinding(**values)


# --------------------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------------------


def test_posts_to_the_version_pinned_action_url(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1", "reward": 1.0}

    with _client(endpoint) as client:
        _rollout(client, "math_rl", "1.0.6", task={"seed": 1}, model=_binding())

    path = recorder.requests[0]["path"]
    assert "/api/projects/proj/rl_environments/math_rl/versions/1.0.6:executeRollout" in path
    assert f"api-version={EXECUTE_ROLLOUT_API_VERSION}" in path


def test_forwards_the_caller_token_in_both_headers(rollout_server):
    """RLE needs ``aml-user-token`` in addition to ``Authorization``; it forwards that one to Loom."""
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1"}

    with _client(endpoint) as client:
        _rollout(client, "math_rl", "1.0.6", task={"seed": 1}, model=_binding())

    headers = recorder.requests[0]["headers"]
    assert headers[FORWARDED_TOKEN_HEADER] == _TOKEN
    assert headers["authorization"] == f"Bearer {_TOKEN}"


def test_sends_task_and_model_binding_verbatim(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1"}
    task = {"problem": "2+2", "nested": {"rows": [1, 2, 3]}}

    with _client(endpoint) as client:
        _rollout(
            client,
            "math_rl",
            "1.0.6",
            task=task,
            model=_binding(renderer_name="qwen3", sequence_id=7),
            rollout_id="fixed-id",
        )

    body = recorder.requests[0]["body"]
    assert body["rollout_id"] == "fixed-id"
    assert body["task"] == task
    assert body["model"] == {
        "model_name": "Qwen/Qwen3-32B",
        "project_endpoint": "https://acct.services.ai.azure.com/api/projects/proj",
        "loom_session_id": "session-1",
        "checkpoint_id": "checkpoint-1",
        "renderer_name": "qwen3",
        "sequence_id": 7,
    }


def test_omits_agent_input_when_unset(rollout_server):
    """Gym/OpenEnv rejects ``agent_input`` outright, so an unset one must not be sent as null."""
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1"}

    with _client(endpoint) as client:
        _rollout(client, "math_rl", "1.0.6", task={"seed": 1}, model=_binding())

    assert "agent_input" not in recorder.requests[0]["body"]


def test_sends_agent_input_for_harness_targets(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1"}

    with _client(endpoint) as client:
        _rollout(
            client,
            "swe",
            "2.0.0",
            task={"row": 1},
            agent_input={"prompt": "fix it"},
            model=_binding(),
        )

    assert recorder.requests[0]["body"]["agent_input"] == {"prompt": "fix it"}


def test_escapes_environment_name_and_version(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1"}

    with _client(endpoint) as client:
        _rollout(client, "odd/name", "1.0.0+beta", task={}, model=_binding())

    path = recorder.requests[0]["path"]
    assert "odd%2Fname/versions/1.0.0%2Bbeta:executeRollout" in path


def test_generates_a_bare_hex_rollout_id_when_omitted():
    body = build_rollout_body(task={}, model=_binding())
    assert len(body.rollout_id) == 32
    assert body.rollout_id == body.rollout_id.lower()
    int(body.rollout_id, 16)


def test_rejects_a_rollout_with_no_task():
    with pytest.raises(ValueError, match="task is required"):
        build_rollout_body(task=None, model=_binding())


@pytest.mark.parametrize(
    "name,version", [("", "1.0.0"), ("math_rl", "")], ids=["no-name", "no-version"]
)
def test_rejects_an_unpinned_environment(rollout_server, name, version):
    _, endpoint = rollout_server
    with _client(endpoint) as client:
        with pytest.raises(ValueError):
            _rollout(client, name, version, task={}, model=_binding())


# --------------------------------------------------------------------------------------
# Response handling
# --------------------------------------------------------------------------------------


def test_returns_reward_episode_and_trainable_sequences(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {
        "rollout_id": "r1",
        "reward": 0.9,
        "rollout": {
            "sequences": [
                {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "logprobs": [-0.1, -0.2, -0.3]}
            ]
        },
        "episode": {
            "kind": "gym_openenv",
            "termination_reason": "environment_terminated",
            "steps": [{"capture_node_id": "n1", "reward": 0.9, "episode_done": True}],
        },
    }

    with _client(endpoint) as client:
        result = _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert result.rollout_id == "r1"
    assert result.reward == 0.9
    assert result.episode.termination_reason == "environment_terminated"
    assert result.episode.steps[0].capture_node_id == "n1"
    assert result.sequences[0]["input_ids"] == [1, 2, 3]


def test_success_stays_absent_for_gym_rather_than_defaulting_to_false(rollout_server):
    """Gym has no grader. A missing verdict must read as unknown, not as a failure."""
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1", "reward": 1.0}

    with _client(endpoint) as client:
        result = _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert result.success is None


def test_sequences_is_empty_for_an_eval_capture(rollout_server):
    """An eval rollout carries reward but no token ids, so there is nothing to train on."""
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1", "reward": 1.0, "rollout": {"sequences": []}}

    with _client(endpoint) as client:
        result = _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert result.sequences == []


def test_sequences_is_empty_when_the_graph_is_absent(rollout_server):
    recorder, endpoint = rollout_server
    recorder.body = {"rollout_id": "r1", "reward": 1.0}

    with _client(endpoint) as client:
        result = _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert result.sequences == []


# --------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------


def test_surfaces_the_service_failure_code_and_rollout_id(rollout_server):
    recorder, endpoint = rollout_server
    recorder.status = 500
    recorder.body = {
        "code": "RolloutDependencyFailed",
        "message": "The Capture Proxy dependency could not complete the rollout "
        "Gym model completion. HTTP 503",
    }

    with _client(endpoint) as client:
        with pytest.raises(RLERolloutError) as caught:
            _rollout(
                client, "math_rl", "1.0.6", task={}, model=_binding(), rollout_id="abc123"
            )

    error = caught.value
    assert error.code == "RolloutDependencyFailed"
    assert error.rollout_id == "abc123"
    assert "HTTP 503" in str(error)
    # The rollout id belongs in the message too: it is the key the service logs the cause under.
    assert "abc123" in str(error)


def test_surfaces_a_nested_error_envelope_from_a_gateway(rollout_server):
    """A gateway ahead of RLE answers in the nested shape, not RLE's flat one."""
    recorder, endpoint = rollout_server
    recorder.status = 403
    recorder.body = {"error": {"code": "AuthorizationFailed", "message": "no data action"}}

    with _client(endpoint) as client:
        with pytest.raises(RLERolloutError) as caught:
            _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert caught.value.code == "AuthorizationFailed"
    assert "no data action" in str(caught.value)


def test_still_raises_when_the_error_body_is_not_json(rollout_server):
    """A failure to parse a failure must not replace it -- the status still has to reach the caller."""
    recorder, endpoint = rollout_server
    recorder.status = 502
    recorder.body = "<html>Bad Gateway</html>"

    with _client(endpoint) as client:
        with pytest.raises(RLERolloutError) as caught:
            _rollout(client, "math_rl", "1.0.6", task={}, model=_binding())

    assert caught.value.status_code == 502
    assert caught.value.code is None
