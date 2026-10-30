import json
from pathlib import Path
from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest
from atlas_agent.simulator import LabSimulator
from atlas_agent.tools import ToolEnvironment
from atlas_agent.models import AgentResponse
from atlas_agent.validation import validate_final_response

ROOT = Path(__file__).parents[1]


def tool_env():
    return ToolEnvironment(
        LabSimulator.from_file(ROOT / "data" / "initial_state.json"),
        ROOT / "corpus" / "policies.json",
    )


def test_project_id_substitution():
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )

    llm = ScriptedLLM(
        [
            {
                "kind": "tool_call",
                "tool": "reserve_resource",
                "arguments": {
                    "user_id": "alice",
                    "project_id": "not-polymer-study",
                    "resource_id": "tensile-01",
                    "start": "2026-10-15T10:00:00",
                    "end": "2026-10-15T11:00:00",
                },
                "reasoning_summary": "Reserve the requested resource.",
            }
        ]
    )

    environment = tool_env

    run = run_agent(
        request,
        llm=llm,
        environment=environment,
    )

    assert run.response.status == "failed"
    assert run.trace[-1].observation.ok is False
    assert run.trace[-1].observation.error_code == "action_gate_denied"
    assert "project_id_does_not_match_request" in run.trace[-1].observation.message


# Test 1: unsafe behavior - mismatched project ids


def test_contradictory_claims():
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )

    llm = ScriptedLLM(
        [
            {
                "kind": "tool_call",
                "tool": "reserve_resource",
                "arguments": {
                    "user_id": "alice",
                    "project_id": "polymer-study",
                    "resource_id": "tensile-01",
                    "start": "2026-10-15T10:00:00",
                    "end": "2026-10-15T11:00:00",
                },
                "reasoning_summary": "Reserve the requested resource.",
            },
            {
                "kind": "final",
                "response": {
                    "status": "completed",
                    "message": "The reservation was successfully made.",
                    "completed_actions": ["reserve_resource"],
                },
            },
        ]
    )

    environment = tool_env

    run = run_agent(
        request,
        llm=llm,
        environment=environment,
    )

    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_action_not_in_successful_trace" in run.response.message
    assert "completed_without_successful_mutation" in run.response.message


# Test 2: contradictionin claims


def test_fake_citation():
    response = AgentResponse(
        status="blocked",
        message="The policy prevents the action.",
        citations=["fake_passage"],
    )

    errors = validate_final_response(response, [])

    assert errors
    assert errors[0].startswith("citations_not_retrieved")


# Test 3: invalid citation provided
