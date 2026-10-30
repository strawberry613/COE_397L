"""Milestone 2: complete the bounded decision–action–observation loop."""

# Ruff suppressions cover names intentionally reserved for the TODOs. Remove
# this line after completing Milestone 2.
## ruff: noqa: F401, F821, F841

from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _failed(message: str, trace: list[TraceEvent], reason: str) -> AgentRun:
    return AgentRun(
        response=AgentResponse(status="failed", message=message),
        trace=trace,
        termination_reason=reason,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    messages = initialize_messages(request)
    trace: list[TraceEvent] = []

    for step in range(1, max_steps + 1):
        try:
            decision = llm.generate(messages=messages, response_model=DecisionAdapter)
            # TODO 2.1 done - model returns a decision
            #  raise NotImplementedError
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            return _failed(
                f"The model did not produce a valid decision: {exc}",
                trace,
                "invalid_model_decision",
            )

        if isinstance(decision, ClarificationDecision):
            return AgentRun(
                response=AgentResponse(
                    status="clarification_required", message=decision.question
                ),
                trace=trace,
                termination_reason="clarification_required",
            )
            # TODO 2.2 done - clarification_required is returned from the run without returning any
            # response or tool to user
        # raise NotImplementedError

        if isinstance(decision, FinalDecision):
            error_list = validate_final_response(
                response=decision.response, trace=trace
            )
            if error_list == []:
                return AgentRun(
                    response=decision.response,
                    trace=trace,
                    termination_reason="model_finished",
                )
            else:
                return _failed(
                    "Inconsistent Final Decision",
                    trace=trace,
                    reason="invalid_model_decision",
                )
            # TODO 2.3 done. Error list checks for inconsistencies and errors based on the validation function.
            # Returns a failure if the error list is populated with any errors
            # raise NotImplementedError

        if not isinstance(decision, ToolCallDecision):
            return _failed(
                "Unsupported decision type.", trace, "invalid_model_decision"
            )

        error_list = validate_proposed_tool_call(decision, request=request, trace=trace)
        if error_list == []:
            observation = environment.dispatch(decision.tool, decision.arguments)
        else:
            observation = ToolObservation(
                ok=False,
                tool=decision.tool,
                error_code="action_gate_denied",
                message="; ".join(error_list),
            )
        # TODO 2.4 done - returns an observation if errors exist, dispatch the tool if there are no errors
        # raise NotImplementedError

        trace.append(
            TraceEvent(
                step=step,
                decision=decision.model_dump(),
                observation=observation,
                mutating=decision.tool in MUTATING_TOOLS,
            )
        )
        # TODO 2.5 done - new trace event added. mutating is defined by MUTATING_TOOLS, not hard coded
        # raise NotImplementedError

        messages.append(
            {
                "role": "assistant",
                "content": decision.model_dump_json(),
            }
        )
        # decsion as assistant message
        messages.append(
            {
                "role": "tool",
                "content": observation.model_dump_json(),
            }
        )
        # observation as a tool message
        # TODO 2.6 done - messages based on decision and tool added
        # raise NotImplementedError

    return _failed("Maximum number of steps exceeded", trace, "maximum_steps_exceeded")
    # TODO 2.7: Return an explicit maximum_steps_exceeded failure.
    # raise NotImplementedError
