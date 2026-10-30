"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:

    error_list = []
    decision_arguments = decision.arguments

    if "user_id" in decision_arguments:
        if decision_arguments["user_id"] != request.authenticated_user_id:
            error_list.append("user_id_must_match_authenticated_user")

    if "project_id" in decision_arguments:
        if request.project_id is None:
            error_list.append("project_id_unresolved_in_request")
        elif decision_arguments["project_id"] != request.project_id:
            error_list.append("project_id_does_not_match_request")

    if "resource_id" in decision_arguments:
        if request.resource_id is None:
            error_list.append("resource_id_unresolved_in_request")
        elif decision_arguments["resource_id"] != request.resource_id:
            error_list.append("resource_id_does_not_match_request")

    if "start" in decision_arguments:
        if request.start is None:
            error_list.append("start_unresolved_in_request")
        elif decision_arguments["start"] != request.start:
            error_list.append("start_does_not_match_request")

    if "end" in decision_arguments:
        if request.end is None:
            error_list.append("end_unresolved_in_request")
        elif decision_arguments["end"] != request.end:
            error_list.append("end_does_not_match_request")

    # """TODO 3.1 - check decision arguments fields to protect agains things like identity substitution."""
    del decision, request
    return error_list


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    error_list = []
    if not context.arguments_complete:
        error_list.append("reservation_arguments_incomplete")

    if not context.policy_retrieved:
        error_list.append("policy_not_retrieved")

    if not context.authorizations_checked:
        error_list.append("authorizations_not_checked")

    if not context.resource_status_checked:
        error_list.append("resource_status_not_checked")

    if not context.project_authorized:
        error_list.append("project_not_authorized")

    if not context.qualification_current:
        error_list.append("required_qualification_missing")

    if not context.resource_available:
        error_list.append("resource_not_available")

    if not context.calibration_current:
        error_list.append("calibration_not_current")

    if not context.maintenance_clear:
        error_list.append("maintenance_not_clear")

    if not context.after_hours_approval_satisfied:
        error_list.append("after_hours_approval_not_satisfied")

    # """TODO 3.2 - check all context fields, return and error is not present."""
    del context
    return error_list


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    error_list = []

    retrieved_citation = retrieved_passage_ids(trace)

    for citation in response.citations:
        if citation not in retrieved_citation:
            error_list.append("citations_not_retrieved")

    sucessful_tool_calls = {
        event.observation.tool for event in trace if event.observation.ok
    }

    for action in response.completed_actions:
        if action not in sucessful_tool_calls:
            error_list.append("completed_action_not_in_successful_trace")

    mutation_event = False
    for event in trace:
        if event.mutating and event.observation.ok:
            mutation_event = True
            break

    if response.status == "completed" and not mutation_event:
        error_list.append("completed_without_successful_mutation")

    # """TODO 3.3 done - check for ciations, tools calls, and mutating events
    del response, trace
    return error_list
