"""Integration tests: a precondition brief actually becomes part of the plan
PGPS reports and the executor runs, not a hidden pre-flight step.

These exercise the real ``data/conditions`` dependency graph (via
``build_condition_model``), exactly like the production pipeline, so a
change to the real createSession / extendQosSessionDuration / verifyLocation
condition data would be caught here too - unlike a fully synthetic
condition model, which could silently drift from what the app actually
ships.
"""

from __future__ import annotations

import pytest

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import get_settings
from camara_pgps.execution.mock_executor import MockExecutor
from camara_pgps.models import Intent
from camara_pgps.models.plan import PlanRequest, PlanStatus
from camara_pgps.models.state import AgentState
from camara_pgps.pgps.conditions_loader import build_condition_model
from camara_pgps.pgps.explain import explain_plan
from camara_pgps.pgps.planner import PGPSPlanner
from camara_pgps.validator.validator import PlanValidator
from camara_pgps.workflow.report import explain


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_settings(get_settings())


@pytest.fixture(scope="module")
def model(registry: CamaraRegistry):
    return build_condition_model(settings=get_settings())


@pytest.fixture(scope="module")
def planner(registry: CamaraRegistry, model) -> PGPSPlanner:
    return PGPSPlanner(registry=registry, condition_model=model)


@pytest.fixture(scope="module")
def validator(registry: CamaraRegistry, model) -> PlanValidator:
    return PlanValidator(registry=registry, condition_model=model)


INTENT = Intent(text="ensure best quality for two hours", goal="ensure best quality", action="ensure")

ALTERNATIVE_BRIEF = {
    "mode": "alternative",
    "resolver_operation": "retrieveSessionsByDevice",
    "if_resolved_operation": "extendQosSessionDuration",
    "if_unresolved_operation": "createSession",
    "reason": "an active session may already exist for this device",
}

ALTERNATIVE_ARGS = {
    "extendQosSessionDuration": {"requestedAdditionalDuration": 7200},
    "createSession": {
        "qosProfile": "QOS_L",
        "applicationServer": {"ipv4Address": "1.2.3.4"},
        "duration": 7200,
    },
}


def _plan_alternative(planner: PGPSPlanner, registry: CamaraRegistry):
    create = registry.require_operation("createSession")
    extend = registry.require_operation("extendQosSessionDuration")
    retrieve = registry.require_operation("retrieveSessionsByDevice")
    request = PlanRequest(
        intent=INTENT,
        candidate_operations=[create, extend, retrieve],
        known_facts={"auth.token": True},
        metadata={"precondition_brief": ALTERNATIVE_BRIEF},
    )
    return planner.plan(request)


class TestPlannerPutsTheCheckInThePlan:
    def test_resolver_is_a_real_first_step(self, planner: PGPSPlanner, registry: CamaraRegistry) -> None:
        plan = _plan_alternative(planner, registry)
        assert plan.status is PlanStatus.VALID
        assert plan.steps[0].operation_id == "retrieveSessionsByDevice"
        assert plan.steps[0].precondition_branch is None

    def test_default_branch_step_is_tagged_and_alt_branch_is_pre_planned(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        plan = _plan_alternative(planner, registry)
        branch_step = plan.steps[1]
        assert branch_step.operation_id == "extendQosSessionDuration"
        branch = branch_step.precondition_branch
        assert branch is not None
        assert branch["mode"] == "alternative"
        assert branch["resolver_operation"] == "retrieveSessionsByDevice"
        # createSession's own sub-chain (it needs a QoS profile) was already
        # fully planned too, in case the check goes the other way.
        alt_ids = [s["operation_id"] for s in branch["alt_steps"]]
        assert alt_ids == ["retrieveQoSProfiles", "createSession"]

    def test_default_branch_gets_a_prior_step_binding_from_the_resolver(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        # Regression guard for the exact gap this feature exists to close:
        # a value the resolver step supplies (qos.session.id -> sessionId)
        # must be recognised as bound, not re-requested from the user.
        plan = _plan_alternative(planner, registry)
        branch_step = plan.steps[1]
        bound_params = {b.parameter for b in branch_step.bindings}
        assert "sessionId" in bound_params

    def test_gate_mode_chains_the_gated_operations_own_prerequisites(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        create = registry.require_operation("createSession")
        verify = registry.require_operation("verifyLocation")
        brief = {
            "mode": "gate",
            "resolver_operation": "verifyLocation",
            "gated_operation": "createSession",
            "reason": "only proceed once location is confirmed",
        }
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[create, verify],
            known_facts={"auth.token": True, "device.identifier": True},
            metadata={"precondition_brief": brief},
        )
        plan = planner.plan(request)
        assert plan.status is PlanStatus.VALID
        assert [s.operation_id for s in plan.steps] == [
            "verifyLocation",
            "retrieveQoSProfiles",
            "createSession",
        ]


class TestPlannerSwapsInARealObservationOperation:
    """The scenario actually seen in real (Ollama-backed) runs: the LLM
    picks a circular resolver (getSession), but a real, already-satisfiable
    observation operation (retrieveSessionsByDevice) sits right next to it
    in the candidate set - precondition_node widens in every sibling
    api_selection rejected, so it's always there for the same API. Falling
    all the way back to ordinary planning in this case is a regression: the
    ordinary solver has no notion of "these are alternatives, only run
    one" and ends up demanding parameters for *both* extendQosSessionDuration
    and createSession at once (e.g. asking for an applicationServer even
    though a session may already exist). The fix should instead keep the
    conditional (check-then-branch) structure and simply swap the resolver.
    """

    CIRCULAR_BRIEF = {
        "mode": "alternative",
        "resolver_operation": "getSession",
        "if_resolved_operation": "extendQosSessionDuration",
        "if_unresolved_operation": "createSession",
        "reason": "check whether a session is already active",
    }

    def _plan(self, planner: PGPSPlanner, registry: CamaraRegistry):
        get_session = registry.require_operation("getSession")
        extend = registry.require_operation("extendQosSessionDuration")
        create = registry.require_operation("createSession")
        retrieve = registry.require_operation("retrieveSessionsByDevice")
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[get_session, extend, create, retrieve],
            known_facts={"auth.token": True},
            metadata={"precondition_brief": self.CIRCULAR_BRIEF},
        )
        return planner.plan(request), request

    def test_resolver_is_swapped_to_a_real_observation_operation(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        plan, _request = self._plan(planner, registry)
        assert plan.status is PlanStatus.VALID
        # retrieveSessionsByDevice - not the LLM's circular getSession - is
        # the real first step, exactly like the non-circular alternative
        # brief case.
        assert plan.steps[0].operation_id == "retrieveSessionsByDevice"
        assert "getSession" not in [s.operation_id for s in plan.steps]

    def test_the_conditional_structure_survives_the_swap(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        # The whole point: this must still be a real check-then-branch plan,
        # not the ordinary-planning fallback that would ask for both
        # branches' parameters up front.
        plan, _request = self._plan(planner, registry)
        branch_step = plan.steps[1]
        assert branch_step.operation_id == "extendQosSessionDuration"
        branch = branch_step.precondition_branch
        assert branch is not None
        assert branch["mode"] == "alternative"
        alt_ids = [s["operation_id"] for s in branch["alt_steps"]]
        assert "createSession" in alt_ids
        # createSession's parameters are only needed if that branch is
        # actually taken at execution time - not demanded up front.

    def test_executor_still_resolves_the_real_branch_after_the_swap(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        plan, request = self._plan(planner, registry)
        validated, _report = validator.validate_and_update(
            plan, known_facts={"auth.token"}, parameters=ALTERNATIVE_ARGS
        )
        assert validated.status is PlanStatus.VALID

        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"retrieveSessionsByDevice": (200, [{"sessionId": "abc-123"}])},
        )
        result = executor.execute(
            validated, arguments=ALTERNATIVE_ARGS, initial_facts={"auth.token": True}, query=INTENT.text
        )
        assert result.succeeded is True
        ran = [s.operation_id for s in result.steps if not s.skipped]
        assert ran == ["retrieveSessionsByDevice", "extendQosSessionDuration"]
        assert "createSession" not in ran
        assert "getSession" not in ran


class TestPlannerRejectsACircularResolver:
    """Regression guard for the exact bug the real Ollama-backed agent hit:
    it chose ``getSession`` (requires ``qos.session.id`` - the very fact a
    resolver is supposed to help discover) as the resolver_operation for an
    alternative brief. Forcing it into step 0 unconditionally used to
    produce a plan whose first step could never be satisfied, which then
    surfaced as an unwanted clarification prompt for ``sessionId`` even
    though the plan "looked" complete. The fix: when an unmet resolver
    requirement has a real producer elsewhere in the graph, the resolver is
    circular and the whole precondition branch is rejected, falling back to
    ordinary planning instead.
    """

    CIRCULAR_BRIEF = {
        "mode": "alternative",
        "resolver_operation": "getSession",
        "if_resolved_operation": "extendQosSessionDuration",
        "if_unresolved_operation": "createSession",
        "reason": "check whether a session is already active",
    }

    def test_circular_resolver_falls_back_to_ordinary_planning(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        get_session = registry.require_operation("getSession")
        extend = registry.require_operation("extendQosSessionDuration")
        create = registry.require_operation("createSession")
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[get_session, extend, create],
            known_facts={"auth.token": True},
            metadata={"precondition_brief": self.CIRCULAR_BRIEF},
        )
        plan = planner.plan(request)
        assert plan.status in (PlanStatus.VALID, PlanStatus.DRAFT)
        step_ids = [s.operation_id for s in plan.steps]
        # The whole point: getSession is never forced into step 0 ahead of
        # its own producer - if it appears at all, the ordinary solver put
        # a real producer of qos.session.id in front of it, exactly as it
        # would for any other candidate operation with an unmet
        # requirement. It must never be the very first step, since nothing
        # precedes it there to supply qos.session.id.
        if "getSession" in step_ids:
            assert step_ids[0] != "getSession"
        # No step is left with an unresolvable *producible* requirement:
        # whatever the ordinary solver produced, a fact some earlier
        # operation could have supplied (e.g. qos.session.id) must actually
        # have been chained in by then - this is the exact gap that used to
        # surface as an unwanted "what is the session ID?" clarification
        # prompt despite the plan "looking" complete. A fact nothing in the
        # graph produces at all (e.g. qos.session.additional_duration, a
        # user-supplied parameter) is a separate, expected kind of "missing"
        # and isn't what this regression guard is about.
        known: set[str] = set(request.known_facts)
        for step in plan.steps:
            missing = set(step.requires) - known
            unchained = [f for f in missing if planner.model.producers_of(f)]
            assert not unchained, f"{step.operation_id} still missing producible fact(s) {unchained}"
            known |= set(step.produces)


class TestPlannerCorrectsABackwardsAlternativeBrief:
    """The exact scenario from a real Ollama-backed run of "Notify me
    whenever I leave the office.": the LLM produced a structurally sound
    alternative brief - resolver=retrieveGeofencingSubscriptionList,
    reason="check whether a geofencing subscription already exists" - but
    assigned the branches backwards. It put createGeofencingSubscription
    (which would redundantly re-create the very subscription the resolver
    just confirmed exists) on the if_resolved side, and the unrelated
    verifyLocation on the if_unresolved side. Rejecting the whole branch
    here would be wrong - the resolver and its reason are sound, only the
    if_resolved/if_unresolved pairing is flipped. The fix swaps them rather
    than giving up on the conditional structure entirely.
    """

    BACKWARDS_BRIEF = {
        "mode": "alternative",
        "resolver_operation": "retrieveGeofencingSubscriptionList",
        "if_resolved_operation": "createGeofencingSubscription",
        "if_unresolved_operation": "verifyLocation",
        "reason": "check whether a geofencing subscription already exists",
    }

    def _plan(self, planner: PGPSPlanner, registry: CamaraRegistry):
        retrieve = registry.require_operation("retrieveGeofencingSubscriptionList")
        create = registry.require_operation("createGeofencingSubscription")
        verify = registry.require_operation("verifyLocation")
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[retrieve, create, verify],
            known_facts={"auth.token": True, "area": True, "config": True, "protocol": True, "sink": True, "types": True},
            metadata={"precondition_brief": self.BACKWARDS_BRIEF},
        )
        return planner.plan(request)

    def test_branches_are_swapped_not_rejected(self, planner: PGPSPlanner, registry: CamaraRegistry) -> None:
        plan = self._plan(planner, registry)
        assert plan.status is PlanStatus.VALID
        assert plan.steps[0].operation_id == "retrieveGeofencingSubscriptionList"

    def test_the_confirmed_branch_no_longer_re_creates_the_subscription(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        # The whole point: once a subscription is confirmed to exist, the
        # taken step must not be the operation that would create a
        # duplicate one.
        plan = self._plan(planner, registry)
        branch_step = plan.steps[1]
        assert branch_step.operation_id == "verifyLocation"
        branch = branch_step.precondition_branch
        assert branch is not None
        assert branch["if_resolved_operation"] == "verifyLocation"
        assert branch["if_unresolved_operation"] == "createGeofencingSubscription"
        alt_ids = [s["operation_id"] for s in branch["alt_steps"]]
        assert "createGeofencingSubscription" in alt_ids


class TestPlannerRejectsAnUnrelatedResolver:
    """The exact scenario from a real Ollama-backed run of "If my SIM
    hasn't been swapped recently, boost my connection for this call.": the
    LLM chose retrieveSimSwapAgeBand (a SIM-swap-recency check, wholly
    unrelated to QoS sessions) as the resolver for an alternative brief
    deciding between extendQosSessionDuration and createSession. Unlike the
    backwards-assignment case above, there is no relationship at all
    between what the resolver produces and what either branch needs or
    produces - this is a fabricated causal link, not a flipped pairing, so
    the correct response is to reject the whole branch and fall back to
    ordinary planning rather than "fix" a pairing that was never sound.
    """

    UNRELATED_BRIEF = {
        "mode": "alternative",
        "resolver_operation": "retrieveSimSwapAgeBand",
        "if_resolved_operation": "extendQosSessionDuration",
        "if_unresolved_operation": "createSession",
        "reason": "check whether the SIM has been swapped recently",
    }

    def test_unrelated_resolver_falls_back_to_ordinary_planning(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        sim_check = registry.require_operation("retrieveSimSwapAgeBand")
        extend = registry.require_operation("extendQosSessionDuration")
        create = registry.require_operation("createSession")
        retrieve = registry.require_operation("retrieveSessionsByDevice")
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[sim_check, extend, create, retrieve],
            known_facts={"auth.token": True},
            metadata={"precondition_brief": self.UNRELATED_BRIEF},
        )
        plan = planner.plan(request)
        assert plan.status in (PlanStatus.VALID, PlanStatus.DRAFT)
        # No trace of the rejected conditional structure should survive:
        # this must be an ordinary plan, not one built around a
        # hallucinated causal link between SIM-swap recency and QoS
        # sessions.
        for step in plan.steps:
            assert step.precondition_branch is None


class TestPlanExplanationShowsBothBranchesBeforeExecution:
    """The whole point of pre-validating both branches (Section 4.7/4.8) is
    lost if nothing shows it before execution - `explain_plan` (called
    right after planning, inside `pgps_node`, before validation or
    execution) is where that has to be visible.
    """

    def test_alternative_mode_shows_both_fully_planned_paths(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        plan = _plan_alternative(planner, registry)
        explanation = explain_plan(
            plan, registry=registry, intent=INTENT, initial_facts={"auth.token"}
        )
        assert len(explanation.branches) == 1
        branch = explanation.branches[0]
        assert branch.resolver_operation == "retrieveSessionsByDevice"
        assert branch.mode == "alternative"
        assert branch.default_path == ["extendQosSessionDuration"]
        # createSession's own sub-chain (a QoS profile lookup) shows up too -
        # this is the fully-planned "other" branch, not just its name.
        assert branch.alternate_path == ["retrieveQoSProfiles", "createSession"]
        assert "Conditional branch(es): 1" in explanation.narrative
        assert "IF NOT confirmed  -> retrieveQoSProfiles -> createSession" in explanation.narrative

    def test_gate_mode_shows_skip_not_a_second_operation(
        self, planner: PGPSPlanner, registry: CamaraRegistry
    ) -> None:
        create = registry.require_operation("createSession")
        verify = registry.require_operation("verifyLocation")
        brief = {
            "mode": "gate",
            "resolver_operation": "verifyLocation",
            "gated_operation": "createSession",
            "reason": "only proceed once location is confirmed",
        }
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[create, verify],
            known_facts={"auth.token": True, "device.identifier": True},
            metadata={"precondition_brief": brief},
        )
        plan = planner.plan(request)
        explanation = explain_plan(
            plan, registry=registry, intent=INTENT,
            initial_facts={"auth.token", "device.identifier"},
        )
        assert len(explanation.branches) == 1
        branch = explanation.branches[0]
        assert branch.mode == "gate"
        assert branch.alternate_path == []
        assert branch.alternate_note and "skipped entirely" in branch.alternate_note
        assert "createSession is skipped entirely" in explanation.narrative

    def test_no_branch_reports_zero(self, planner: PGPSPlanner, registry: CamaraRegistry) -> None:
        extend = registry.require_operation("extendQosSessionDuration")
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[extend],
            known_facts={"auth.token": True, "qos.session.id": True},
        )
        plan = planner.plan(request)
        explanation = explain_plan(plan, registry=registry, intent=INTENT)
        assert explanation.branches == []
        assert "Conditional branch(es): 0" in explanation.narrative


class TestReportShowsTheAlternateBranchHonestly:
    """report.py's explain() must not lump the pre-validated alternate
    branch (createSession, held in reserve behind the session-exists check)
    in with genuinely irrelevant candidates (checkSimSwap, verifyLocation)
    under "Rejected candidate(s)" - doing so is exactly what made a real,
    working conditional plan look, in the CLI narrative, like it had
    silently discarded its own second branch.
    """

    class _FakeSelection:
        def __init__(self, rejected):
            self.apis = []
            self.operations = []
            self.rejected = rejected

    def test_the_untaken_branch_is_reported_as_pre_validated_not_rejected(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        plan = _plan_alternative(planner, registry)
        validated, _report = validator.validate_and_update(
            plan, known_facts={"auth.token"}, parameters=ALTERNATIVE_ARGS
        )
        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"retrieveSessionsByDevice": (200, [{"sessionId": "abc-123"}])},
        )
        result = executor.execute(
            validated, arguments=ALTERNATIVE_ARGS, initial_facts={"auth.token": True}, query=INTENT.text
        )

        state = AgentState(
            request_text=INTENT.text,
            intent=INTENT,
            selection=self._FakeSelection(
                rejected=["createSession", "checkSimSwap", "verifyLocation"]
            ),
            selected_operations=[
                registry.require_operation(op_id)
                for op_id in ("retrieveSessionsByDevice", "extendQosSessionDuration", "createSession")
            ],
            plan=validated,
            execution=result,
        )

        narrative = explain(state)
        assert "createSession" in narrative
        assert "pre-validated as the alternate branch" in narrative
        # The real point: createSession must not also show up in the
        # ordinary "Rejected candidate(s)" bucket, alongside operations
        # that were never relevant to this request at all.
        rejected_line = next(
            (line for line in narrative.split(". ") if line.startswith("Rejected candidate")),
            "",
        )
        assert "createSession" not in rejected_line
        assert "checkSimSwap" in rejected_line
        assert "verifyLocation" in rejected_line


class TestExecutorResolvesTheRealBranch:
    def _validated(self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry):
        plan = _plan_alternative(planner, registry)
        validated, _report = validator.validate_and_update(
            plan, known_facts={"auth.token"}, parameters=ALTERNATIVE_ARGS
        )
        assert validated.status is PlanStatus.VALID
        return validated

    def test_session_found_runs_extend_for_real(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        validated = self._validated(planner, validator, registry)
        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"retrieveSessionsByDevice": (200, [{"sessionId": "abc-123"}])},
        )
        result = executor.execute(
            validated, arguments=ALTERNATIVE_ARGS, initial_facts={"auth.token": True}, query=INTENT.text
        )
        assert result.succeeded is True
        ran = [s.operation_id for s in result.steps if not s.skipped]
        assert ran == ["retrieveSessionsByDevice", "extendQosSessionDuration"]
        outcome = result.metadata["precondition_outcomes"][0]
        assert outcome["resolved"] is True
        assert outcome["ran_operation"] == "extendQosSessionDuration"

    def test_no_session_splices_in_the_full_createSession_chain(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        validated = self._validated(planner, validator, registry)
        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"retrieveSessionsByDevice": (200, [])},
        )
        result = executor.execute(
            validated, arguments=ALTERNATIVE_ARGS, initial_facts={"auth.token": True}, query=INTENT.text
        )
        assert result.succeeded is True
        ran = [s.operation_id for s in result.steps if not s.skipped]
        # the alternate branch's own prerequisite (a QoS profile lookup) was
        # never in the tentative plan at all - it only exists because the
        # executor spliced in createSession's real sub-chain.
        assert ran == ["retrieveSessionsByDevice", "retrieveQoSProfiles", "createSession"]
        outcome = result.metadata["precondition_outcomes"][0]
        assert outcome["resolved"] is False
        assert outcome["ran_operation"] == "createSession"

    def test_gate_not_confirmed_skips_the_gated_operation_without_calling_it(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        create = registry.require_operation("createSession")
        verify = registry.require_operation("verifyLocation")
        brief = {
            "mode": "gate",
            "resolver_operation": "verifyLocation",
            "gated_operation": "createSession",
            "reason": "only proceed once location is confirmed",
        }
        args = {
            "verifyLocation": {
                "area": {
                    "areaType": "CIRCLE",
                    "center": {"latitude": 50.1109, "longitude": 8.6821},
                    "radius": 5000,
                }
            },
            "createSession": {
                "qosProfile": "QOS_L",
                "applicationServer": {"ipv4Address": "1.2.3.4"},
                "duration": 3600,
            },
        }
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[create, verify],
            known_facts={"auth.token": True, "device.identifier": True},
            metadata={"precondition_brief": brief},
        )
        plan = planner.plan(request)
        validated, _report = validator.validate_and_update(
            plan, known_facts={"auth.token", "device.identifier"}, parameters=args
        )
        assert validated.status is PlanStatus.VALID

        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"verifyLocation": (200, {"verificationResult": "FALSE"})},
        )
        result = executor.execute(
            validated, arguments=args, initial_facts={"auth.token": True, "device.identifier": True}, query=INTENT.text
        )
        assert result.succeeded is True  # correctly declining to run is not a failure
        assert [s.operation_id for s in result.steps if not s.skipped] == ["verifyLocation"]
        skipped = [s for s in result.steps if s.skipped]
        assert len(skipped) == 1
        outcome = result.metadata["precondition_outcomes"][0]
        assert outcome["resolved"] is False
        assert outcome["ran_operation"] is None
        assert outcome["skipped_operation"] == "createSession"

    def test_gate_confirmed_runs_the_gated_operations_full_chain(
        self, planner: PGPSPlanner, validator: PlanValidator, registry: CamaraRegistry, model
    ) -> None:
        create = registry.require_operation("createSession")
        verify = registry.require_operation("verifyLocation")
        brief = {
            "mode": "gate",
            "resolver_operation": "verifyLocation",
            "gated_operation": "createSession",
            "reason": "only proceed once location is confirmed",
        }
        args = {
            "verifyLocation": {
                "area": {
                    "areaType": "CIRCLE",
                    "center": {"latitude": 50.1109, "longitude": 8.6821},
                    "radius": 5000,
                }
            },
            "createSession": {
                "qosProfile": "QOS_L",
                "applicationServer": {"ipv4Address": "1.2.3.4"},
                "duration": 3600,
            },
        }
        request = PlanRequest(
            intent=INTENT,
            candidate_operations=[create, verify],
            known_facts={"auth.token": True, "device.identifier": True},
            metadata={"precondition_brief": brief},
        )
        plan = planner.plan(request)
        validated, _report = validator.validate_and_update(
            plan, known_facts={"auth.token", "device.identifier"}, parameters=args
        )
        assert validated.status is PlanStatus.VALID

        executor = MockExecutor(
            registry=registry,
            condition_model=model,
            responses={"verifyLocation": (200, {"verificationResult": "TRUE"})},
        )
        result = executor.execute(
            validated, arguments=args, initial_facts={"auth.token": True, "device.identifier": True}, query=INTENT.text
        )
        assert result.succeeded is True
        assert [s.operation_id for s in result.steps] == [
            "verifyLocation",
            "retrieveQoSProfiles",
            "createSession",
        ]
        outcome = result.metadata["precondition_outcomes"][0]
        assert outcome["resolved"] is True
        assert outcome["ran_operation"] == "createSession"
