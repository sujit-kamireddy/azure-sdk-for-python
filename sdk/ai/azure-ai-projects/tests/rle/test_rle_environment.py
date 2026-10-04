# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

from typing import Any, Optional

import pytest

pytest.importorskip("openenv")
pytest.importorskip("fastmcp")

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment  # type: ignore[import-untyped]
from openenv.core.env_server.mcp_types import (  # type: ignore[import-untyped]
    CallToolAction,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation, State  # type: ignore[import-untyped]

from azure.ai.projects.rle.environments import GradeAction, RLEnvironment


class _TestEnvironment(RLEnvironment):
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        self._set_state(State(episode_id=episode_id))
        return Observation(metadata={"seed": seed, **kwargs})

    def grade(
        self,
        action: GradeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        current_state = self.state
        self._set_state(
            current_state.model_copy(
                update={"step_count": current_state.step_count + 1}
            )
        )
        return Observation(
            reward=1.0 if action.answer == "correct" else 0.0,
            done=True,
            metadata={"timeout_s": timeout_s, **kwargs},
        )


def test_public_environment_inherits_openenv_mcp_environment() -> None:
    assert issubclass(RLEnvironment, MCPEnvironment)


def test_subclass_must_implement_reset_and_grade() -> None:
    assert RLEnvironment.__abstractmethods__ == frozenset({"grade", "reset"})

    class MissingImplementations(RLEnvironment):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingImplementations()


def test_grade_action_requires_string_answer() -> None:
    with pytest.raises(ValueError):
        GradeAction()  # type: ignore[call-arg]

    with pytest.raises(ValueError):
        GradeAction(answer=1)  # type: ignore[arg-type]


def test_grade_action_rollout_defaults_to_none() -> None:
    assert GradeAction(answer="correct").rollout is None


def test_grade_action_accepts_rollout_graph() -> None:
    rollout = {"turns": [{"tool_call_errors": [{"error_code": "invalid_json"}]}]}

    action = GradeAction(answer="correct", rollout=rollout)

    assert action.rollout == rollout


def test_grade_action_dispatches_to_grade() -> None:
    environment = _TestEnvironment()

    observation = environment.step(
        GradeAction(answer="correct"), timeout_s=5.0, attempt=1
    )

    assert observation.reward == 1.0
    assert observation.done is True
    assert observation.metadata == {"timeout_s": 5.0, "attempt": 1}
    assert environment.state.step_count == 1


def test_non_mcp_action_must_be_grade_action() -> None:
    environment = _TestEnvironment()

    with pytest.raises(TypeError, match="Expected GradeAction"):
        environment.step(Action())


def test_environment_can_have_no_custom_mcp_tools() -> None:
    environment = _TestEnvironment()

    observation = environment.step(ListToolsAction())

    assert observation.tools == []


def test_tool_decorator_registers_callable_mcp_tool() -> None:
    environment = _TestEnvironment()

    @environment.tool()
    def echo(value: str) -> str:
        return value

    listed = environment.step(ListToolsAction())
    called = environment.step(
        CallToolAction(tool_name="echo", arguments={"value": "hello"})
    )

    assert [tool.name for tool in listed.tools] == ["echo"]
    assert called.error is None
    assert "hello" in str(called.result)


def test_preconfigured_mcp_server_preserves_registered_tools() -> None:
    server = FastMCP("preconfigured")

    @server.tool()
    def lookup(value: str) -> str:
        return value

    environment = _TestEnvironment(server)

    observation = environment.step(ListToolsAction())

    assert [tool.name for tool in observation.tools] == ["lookup"]


def test_state_returns_defensive_snapshot() -> None:
    environment = _TestEnvironment()
    environment.reset(episode_id="episode-1")

    snapshot = environment.state
    snapshot.step_count = 10

    assert environment.state.episode_id == "episode-1"
    assert environment.state.step_count == 0


def test_set_state_rejects_non_openenv_state() -> None:
    environment = _TestEnvironment()

    with pytest.raises(TypeError, match="OpenEnv State"):
        environment._set_state(object())  # type: ignore[arg-type]
