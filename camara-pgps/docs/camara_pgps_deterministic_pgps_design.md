# CAMARA-PGPS: Generic Deterministic Precondition-Guided Plan Synthesis

## 1. Purpose

This document defines the target architecture of **CAMARA-PGPS** as a generic, deterministic planning framework for CAMARA API orchestration.

The central thesis contribution is:

> **Precondition-Guided Plan Synthesis (PGPS): a deterministic, goal-oriented planning framework that transforms grounded user goals and CAMARA API capabilities into complete, dependency-aware, conditionally executable plans by reasoning over explicit preconditions, effects, observations, and state transitions.**

The planner is deliberately **LLM-free**.

LLMs may be used by other parts of the system for intent understanding, multi-intent decomposition, semantic retrieval, API/capability selection, parameter extraction, natural-language item disambiguation, and optional explanation generation.

However, the authoritative decisions about which operations are required, dependencies, ordering, goal achievement, branch construction, branch feasibility, and plan validity belong to PGPS and deterministic validation.

---

# 2. Core Design Principle

The system follows a strict separation of concerns:

```text
Natural Language
      |
      v
+-----------------------------+
| LLM-Based Understanding     |
|                             |
| Intent decomposition        |
| Retrieval                   |
| API selection               |
| Parameter extraction        |
+--------------+--------------+
               |
               | grounded structured input
               v
+---------------------------------------------+
|                 PGPS                        |
|                                             |
| Goal reasoning                              |
| Preconditions / effects                     |
| Dependency resolution                       |
| State reasoning                             |
| Observation discovery                      |
| Conditional plan synthesis                 |
| Multi-intent planning                      |
+----------------------+----------------------+
                       |
                       v
+---------------------------------------------+
| Deterministic Plan Validation               |
|                                             |
| Structure, dependencies, parameters,       |
| payload/schema, branch completeness         |
+----------------------+----------------------+
                       |
                       | EXECUTION BARRIER
                       v
+---------------------------------------------+
| Runtime Execution                           |
|                                             |
| API execution                               |
| Observation                                 |
| World-state update                          |
| Prevalidated branch selection               |
+---------------------------------------------+
```

The key boundary is:

> **LLMs propose; PGPS decides; validation disposes; runtime observes and executes.**

PGPS itself must not call an LLM.

---

# 3. Determinism Definition

The entire application is not deterministic because upstream components may use LLMs.

The precise thesis claim is:

> **PGPS planning and deterministic validation are deterministic functions of their normalized structured inputs, condition model, goals, and fixed planning policy.**

Conceptually:

```text
Plan = PGPS(
    Goals,
    InitialState,
    Operations,
    OperationConditions,
    IntentGraph,
    PlanningPolicy
)
```

For identical inputs, PGPS must return the same result.

Therefore PGPS must contain:

- no LLM calls;
- no randomness;
- no temperature-dependent decisions;
- no arbitrary producer selection;
- no dependence on set/dictionary iteration order;
- no API-specific planning branches;
- no hidden network calls;
- no runtime observations required to construct the initial plan.

Deterministic tie-breaking must be explicit. A possible policy is:

```text
1. goal relevance
2. applicability
3. number of satisfied required conditions
4. additional dependency cost
5. side-effect cost
6. observation cost
7. execution cost
8. canonical operation_id ordering
```

The policy can be configurable, but a fixed configuration produces reproducible results.

---

# 4. What Is Already Good and Must Be Preserved

The current implementation contains several strong architectural decisions. They should be preserved.

## 4.1 LLM / PGPS separation

The current split between upstream LLM reasoning and deterministic planning is correct.

Keep:

```text
LLM -> structured proposals
PGPS -> authoritative planning
Validator -> authoritative validity
```

An LLM must never directly modify the final execution graph.

---

## 4.2 Registry and planning knowledge are separate

The existing separation between CAMARA OpenAPI information and planning knowledge is correct.

### OpenAPI/RAG knowledge

Answers:

```text
Does the operation exist?
What HTTP method does it use?
What path does it use?
What parameters exist?
What request body is required?
What response does it return?
What schema is used?
```

### PGPS condition knowledge

Answers:

```text
What must be true before the operation?
What facts does it establish?
What facts does it invalidate?
What observations can establish runtime state?
What dependencies exist?
```

OpenAPI should not be forced to contain planning semantics it does not provide.

---

## 4.3 Declarative condition files

Planning semantics remain declarative:

```text
data/conditions/*.conditions.json
```

Adding a new API should normally require data changes rather than planner-code changes.

This is essential for genericity.

---

## 4.4 `produces_from`

The current declarative response-to-fact mapping is a strong design and must remain.

Example:

```json
{
  "produces": ["qos.session.id"],
  "produces_from": {
    "qos.session.id": "sessionId"
  }
}
```

The executor should not contain normal operation-specific extraction such as:

```python
if operation_id == "...":
    ...
```

New operations should be supported through condition metadata.

---

## 4.5 Canonical parameter/fact vocabulary

Keep `pgps/parameter_facts.py`.

The vocabulary should connect:

```text
user-extracted values
        |
        v
facts
        |
        v
parameter bindings
        |
        v
API parameters
```

This prevents separate modules from maintaining incompatible mappings.

---

## 4.6 Plan steps carry planning semantics

`PlanStep` should continue to carry the actual semantics used to create it:

```text
requires
produces
invalidates
updates
bindings
reason
```

Downstream components should consume the same plan representation rather than independently rebuilding the dependency model.

---

## 4.7 PGPS can discover prerequisites beyond API selection

API selection should provide grounded candidates, not dictate the final plan.

Example:

```text
Goal:
extend my gaming QoS session

Candidate:
extendQosSessionDuration
```

PGPS reasons:

```text
extendQosSessionDuration
    requires qos.session.id

qos.session.id
    is produced by retrieveSessionsByDevice
```

Therefore:

```text
retrieveSessionsByDevice
        ->
extendQosSessionDuration
```

This must work without the planner knowing either operation name in advance.

---

## 4.8 Prior-step parameter bindings

If:

```text
Step 1 produces qos.session.id
Step 2 requires qos.session.id
```

then PGPS generates:

```text
Step 2.sessionId <- PRIOR_STEP(qos.session.id)
```

This generic fact-based binding mechanism should remain.

---

## 4.9 Complete branch pre-validation

This is one of the strongest current architectural properties.

The system must synthesize and validate all reachable branches before real side-effecting execution.

```text
                 Observation
                      |
             +--------+--------+
             |                 |
           TRUE              FALSE
             |                 |
         Branch A          Branch B
```

Both branches are validated before the execution barrier.

---

# 5. Target Architecture

```text
                         USER REQUEST
                              |
                              v
                  +------------------------+
                  | Intent Decomposer      |
                  |        LLM             |
                  +-----------+------------+
                              |
                              v
                       Intent Graph
                              |
                              v
                  +------------------------+
                  | Goal Construction      |
                  | / Grounding            |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | RAG Retrieval          |
                  |        LLM/RAG         |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | API Selection          |
                  |        LLM             |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  | Parameter Extraction   |
                  |        LLM/rules       |
                  +-----------+------------+
                              |
                              |
             =====================================
                    DETERMINISTIC BOUNDARY
             =====================================
                              |
                              v
                  +------------------------+
                  |        PGPS            |
                  |        NO LLM           |
                  |                        |
                  | Explicit Goals         |
                  | World State            |
                  | Predicates             |
                  | Effects                |
                  | Dependencies           |
                  | Observations           |
                  | Goal-oriented search   |
                  | Multi-intent planning  |
                  | Conditional branches   |
                  +-----------+------------+
                              |
                              v
                  Complete Conditional Plan
                              |
                              v
                  +------------------------+
                  | Deterministic Validator|
                  +-----------+------------+
                              |
                    +---------+---------+
                    |                   |
                  INVALID              VALID
                    |                   |
                   STOP          EXECUTION BARRIER
                                        |
                                        v
                              +----------------+
                              | API Executor   |
                              +-------+--------+
                                      |
                                      v
                                Runtime State
                                      |
                                      v
                              Branch Selection
                                      |
                         +------------+------------+
                         |                         |
                    Prevalidated A            Prevalidated B
                         |                         |
                      Execute                   Execute
```

---

# 6. First-Class Goal Representation

The current implicit goal derivation from the union of candidate preconditions is useful as a compatibility mechanism, but it should not be the final planning semantics.

Introduce:

```python
class Goal(BaseModel):
    id: str
    description: str
    conditions: list[Condition]
    priority: int = 0
    intent_id: str | None = None
    metadata: dict[str, Any] = {}
```

A goal represents a desired state or outcome.

Examples:

```text
qos.session.exists == TRUE
```

```text
qos.session.duration.updated == TRUE
```

```text
notification.subscription.exists == TRUE
```

The goal is not simply "execute operation X."

---

# 7. Goal-Achieving vs Supporting Operations

Do not force-include every API-selection candidate.

Classify operations conceptually as:

```text
Goal-achieving operation
Supporting operation
Observation operation
Irrelevant operation
```

Example:

```text
Goal:
extend QoS session

Candidates:
extendSession
retrieveSessionsByDevice
retrieveQoSProfiles
checkSimSwap
```

If only the first two are required, the plan must not contain the other two.

The planner should produce the **minimum sufficient plan** under its deterministic planning policy.

---

# 8. Sink Operations

The current force-inclusion mechanism solved an important sink-operation problem but creates plan bloat.

A sink operation may produce no reusable fact while being the actual requested action.

Example:

```text
extendQosSessionDuration
```

Therefore represent goal effects explicitly.

Example:

```json
{
  "operation_id": "extendQosSessionDuration",
  "goal_effects": [
    "qos.session.duration.updated"
  ]
}
```

or define the goal directly as:

```text
qos.session.duration.updated == TRUE
```

This allows backward planning to discover the sink operation through the goal instead of force-including all selected candidates.

---

# 9. World State

Replace positive-key-only `FactSet` semantics with an explicit `WorldState`.

## 9.1 Three-valued knowledge

A fact can be:

```text
TRUE
FALSE
UNKNOWN
```

Example:

```text
qos.session.exists = UNKNOWN
```

does not mean false.

It means the planner does not know.

---

## 9.2 Conceptual model

```python
class TruthValue(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
```

```python
class StateValue(BaseModel):
    truth: TruthValue
    value: Any | None = None
    source: str | None = None
```

```python
class WorldState(BaseModel):
    facts: dict[str, StateValue]
```

Examples:

```text
device.id = TRUE("+491234567")

duration = TRUE("30m")

qos.session.id = UNKNOWN

qos.session.exists = UNKNOWN
```

---

# 10. Predicate-Based Conditions

A fact key alone is insufficient for conditional planning.

Introduce:

```python
class Operator(str, Enum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
```

Condition:

```python
class Condition(BaseModel):
    fact: str
    operator: Operator
    value: Any | None = None
```

Examples:

```json
{
  "fact": "qos.session.exists",
  "operator": "eq",
  "value": true
}
```

```json
{
  "fact": "qos.session.status",
  "operator": "eq",
  "value": "ACTIVE"
}
```

```json
{
  "fact": "qos.profile.priority",
  "operator": "lte",
  "value": 10
}
```

---

# 11. Condition Evaluation

Evaluating a condition produces:

```text
SATISFIED
UNSATISFIED
UNKNOWN
```

For example:

```text
Condition:
qos.session.exists == TRUE

State:
qos.session.exists = UNKNOWN

Result:
UNKNOWN
```

This is different from:

```text
qos.session.exists = FALSE
```

which is:

```text
UNSATISFIED
```

---

# 12. Effects and State Transitions

The current monotonic `apply(produces)` model should be expanded.

Operations should support:

```text
produces
updates
invalidates
```

Example:

```json
{
  "operation_id": "createSession",
  "kind": "action",
  "produces": [
    "qos.session.exists",
    "qos.session.id",
    "qos.session.status"
  ],
  "updates": {
    "qos.session.exists": true,
    "qos.session.status": "ACTIVE"
  },
  "produces_from": {
    "qos.session.id": "sessionId"
  }
}
```

Delete:

```json
{
  "operation_id": "deleteSession",
  "kind": "action",
  "invalidates": [
    "qos.session.exists",
    "qos.session.status",
    "qos.session.id"
  ],
  "updates": {
    "qos.session.exists": false
  }
}
```

The planner reasons over:

```text
State_before
      |
      | operation
      v
State_after
```

---

# 13. Observation Operations

Make observations first-class.

```json
{
  "operation_id": "retrieveSessionsByDevice",
  "kind": "observation"
}
```

Actions:

```json
{
  "operation_id": "createSession",
  "kind": "action"
}
```

An observation:

- reads backend state;
- does not intentionally modify the target resource;
- produces facts used for decision making.

This allows PGPS to mechanically distinguish:

```text
"check whether something exists"
```

from:

```text
"create something"
```

---

# 14. Observation Selection Must Be Deterministic

Given:

```text
qos.session.exists = UNKNOWN
```

PGPS finds operations that can establish that fact.

It should require:

```text
kind == observation
```

Then evaluate their own preconditions.

If:

```text
retrieveSessionsByDevice
```

requires only:

```text
device.id
```

and `device.id` is known, it is a valid observation candidate.

If:

```text
getSession
```

requires:

```text
qos.session.id
```

then it cannot be the initial resolver for an unknown session ID.

This removes the circular-resolver problem generically.

---

# 15. Conditional Plan as a First-Class Model

Do not represent conditional logic only as an unstructured dictionary attached to a plan step.

Use:

```python
class PlanBranch(BaseModel):
    id: str
    condition: Condition
    steps: list[PlanStep]
    validation: ValidationResult
```

```python
class ConditionalPlan(BaseModel):
    observation_step: PlanStep
    branches: list[PlanBranch]
    common_steps: list[PlanStep] = []
    validation: ValidationResult
```

Conceptually:

```text
ConditionalPlan
 |
 +-- Observation
 |
 +-- Branch TRUE
 |     |
 |     +-- Step 1
 |     +-- Step 2
 |
 +-- Branch FALSE
       |
       +-- Step 1
       +-- Step 2
```

---

# 16. Conditional Planning

Example:

```text
Goal:
ensure a QoS session exists and extend it
```

PGPS creates:

```text
retrieveSession
       |
       | session.exists == TRUE
       v
extendSession
```

and:

```text
retrieveSession
       |
       | session.exists == FALSE
       v
createSession
       |
       v
extendSession
```

Both paths are planned before execution.

---

# 17. Branch Completeness

A conditional plan is complete only when every relevant outcome of the observation is covered.

For a boolean observation:

```text
TRUE
FALSE
```

both paths need a defined continuation.

For multi-valued state:

```text
ACTIVE
EXPIRED
NOT_FOUND
```

the condition model must explicitly describe how each relevant state is handled.

Possible handling:

```text
branch
terminate safely
return error
mark infeasible
```

The planner must never silently ignore a reachable outcome.

---

# 18. Execution Barrier

The execution barrier is a core safety invariant:

```text
                 PGPS
                   |
                   v
          Complete Conditional Plan
                   |
                   v
        Validate ALL reachable branches
                   |
          +--------+--------+
          |                 |
        invalid             valid
          |                 |
         STOP         EXECUTION BARRIER
                            |
                            v
                       Real API calls
```

Rule:

> **No side-effecting operation executes until all required reachable branches have passed deterministic validation.**

Therefore:

> **No branch is invented after execution begins.**

---

# 19. Runtime Branch Selection

Runtime does not plan.

Runtime does:

```text
execute observation
        |
        v
extract facts
        |
        v
update WorldState
        |
        v
evaluate branch predicates
        |
        v
select already validated branch
        |
        v
execute branch
```

Runtime must not:

```text
ask an LLM what to do next
```

or:

```text
invoke PGPS to invent a new branch
```

for normal conditional execution.

---

# 20. Static Validation

Validation should have separate layers.

## 20.1 Structural validation

Check:

```text
operation exists
dependencies valid
preconditions resolvable
no cycles
branch references valid
observation can establish decision fact
goal reachable
```

## 20.2 Parameter validation

Check:

```text
required parameter available
parameter has valid source
prior-step binding exists where needed
no unresolved required parameter
```

## 20.3 Payload/schema validation

Check:

```text
types
required fields
request-body structure
enum values
object/array shape
OpenAPI schema compatibility
```

This prevents a value such as:

```text
"office"
```

from being accepted as a structured geofencing `area` merely because some value was extracted.

## 20.4 Branch validation

Every required branch is independently validated.

```text
TRUE branch:
VALID

FALSE branch:
VALID
```

Only then:

```text
ConditionalPlan:
VALID
```

---

# 21. Deterministic Dependency Graph

Build the dependency graph from formal conditions:

```text
Operation -> requires -> Fact
Operation -> produces -> Fact
```

Therefore:

```text
Producer A -> Fact X -> Consumer B
```

creates:

```text
A -> B
```

The graph contains no semantic guesses.

---

# 22. Deterministic Producer Selection

Remove:

```python
producers[0]
```

If several operations produce the same fact:

```text
A -> session.id
B -> session.id
C -> session.id
```

evaluate all candidates.

Possible deterministic ranking:

```text
1. applicable from current state
2. fewer unresolved preconditions
3. fewer dependencies
4. directly contributes to goal
5. lower planning cost
6. observation preferred over unnecessary side effect
7. canonical operation_id tie-break
```

This is generic and independent of API names.

---

# 23. Planning Cost

Optionally define:

```python
class PlanningCost(BaseModel):
    dependency_cost: float = 0
    side_effect_cost: float = 0
    observation_cost: float = 0
    execution_cost: float = 0
```

Then use a fixed policy.

This allows deterministic selection between multiple valid producers.

---

# 24. Goal-Oriented Search

The planner should search from goals rather than from all candidates.

Conceptually:

```text
Goal
  ^
  |
effects
  ^
  |
candidate operations
  ^
  |
preconditions
```

At the same time, symbolic state is propagated forward:

```text
InitialState
     |
     v
Operation
     |
     v
State'
```

Thus:

> **Backward search identifies what is required; forward state propagation verifies what becomes true.**

---

# 25. Minimum Sufficient Plan

The objective should be:

> Construct the smallest valid operation set that satisfies all goals and required branch conditions under the configured deterministic planning policy.

This fixes the current force-inclusion bloat while preserving sink operations through explicit goal effects.

Do not allow the number of candidates returned by API selection to determine final plan size.

---

# 26. Dominated Plan Removal

If two plans satisfy the same goal:

```text
A -> B -> Goal
```

and:

```text
C -> D -> E -> Goal
```

and the first has lower deterministic cost while satisfying the same constraints, the second can be discarded.

This is an optimization, not an LLM decision.

---

# 27. Multi-Intent Planning

Multi-intent must be represented explicitly.

Example:

```text
User:
Find my gaming QoS profile and then extend my active session by 30 minutes.
```

LLM decomposition:

```text
Intent 1:
Find gaming QoS profile

Intent 2:
Extend active session by 30 minutes
```

Dependency:

```text
Intent 1 -> Intent 2
```

if Intent 2 needs the result of Intent 1.

---

# 28. Intent Graph

Introduce:

```python
class IntentNode(BaseModel):
    id: str
    goal: Goal
    parameters: dict[str, Any]
```

```python
class IntentDependency(BaseModel):
    source_intent: str
    target_intent: str
    required_facts: list[str]
```

```python
class IntentGraph(BaseModel):
    intents: list[IntentNode]
    dependencies: list[IntentDependency]
```

PGPS consumes this structure without needing an LLM.

---

# 29. Multi-Intent Planning Algorithm

For each intent:

```text
Goal_i
  |
  v
required subplan
```

Then merge plans using shared facts.

Example:

```text
Intent 1:
retrieve profile
      |
      | qos.profile.id
      v
Intent 2:
create session
      |
      | qos.session.id
      v
Intent 3:
extend session
```

Final plan:

```text
retrieve profile
      ->
create session
      ->
extend session
```

---

# 30. Independent Intents

If two intents are unrelated:

```text
Intent A -> Plan A

Intent B -> Plan B
```

do not manufacture:

```text
A -> B
```

The planner should preserve independence.

This directly prevents false causal relationships between unrelated API domains.

---

# 31. Shared Facts Across Intents

If:

```text
Intent A produces device.id
Intent B requires device.id
```

reuse the fact.

Do not perform redundant discovery.

Conceptually:

```text
discover device
      |
      +------> Intent A
      |
      +------> Intent B
```

---

# 32. Global Multi-Intent Goal

The planning problem becomes:

```text
GlobalGoal = {Goal_1, Goal_2, ..., Goal_n}
```

PGPS must satisfy all required goals and intent dependencies.

Independent goals remain independent.

Shared dependencies are reused.

---

# 33. LLM Authority Boundary

Allowed LLM components:

```text
Intent Decomposer
RAG query generation
API Selection
Parameter Extraction
Item Selection
Optional explanation generation
```

Not allowed inside PGPS:

```text
Goal validation
Dependency graph
Producer selection
Precondition evaluation
Branch construction
Cycle detection
Plan validation
Payload validation
Runtime branch selection
```

---

# 34. LLMs Cannot Create Causal Relationships

If an LLM proposes:

```text
checkSimSwap -> createSession
```

PGPS verifies:

```text
Does checkSimSwap produce a fact required by createSession?
```

If not:

```text
INVALID RELATIONSHIP
```

The proposal is discarded.

The condition model is authoritative.

---

# 35. Candidate Operations vs Authoritative Plan

API Selection produces:

```text
candidate capabilities
```

PGPS produces:

```text
authoritative operation set
```

The planner may search beyond selected candidates for supporting operations when the condition model proves they are needed.

Recommended policy:

```text
goal-achieving operations
+
deterministic prerequisite closure
```

not:

```text
all selected candidates
```

---

# 36. Genericity Requirement

A core thesis requirement should be:

> **Adding a new CAMARA API must not require changing PGPS planner code merely because the API is new.**

Expected process:

```text
OpenAPI
   |
   v
normalized operation metadata
   |
   v
declarative planning conditions
   |
   v
generic PGPS
```

No new planner branch should be needed.

---

# 37. No Operation-Specific Logic

Bad:

```python
if operation_id == "retrieveSessionsByDevice":
    ...
elif operation_id == "createSession":
    ...
```

Good:

```text
read conditions
find producer
evaluate preconditions
construct dependency
propagate state
continue
```

PGPS must reason over structure, not names.

---

# 38. Generic Synthetic Example

A non-CAMARA test can use:

```text
operation_A:
    requires x
    produces y

operation_B:
    requires y
    produces z

operation_C:
    requires z
    produces goal
```

PGPS must return:

```text
A -> B -> C
```

If the names are replaced by arbitrary IDs, behavior must remain identical.

This is a strong test of genericity.

---

# 39. Generic Conditional Example

```text
observe_resource:
    kind = observation
    produces resource.exists

create_resource:
    kind = action
    requires NOT resource.exists
    produces resource.exists

update_resource:
    kind = action
    requires resource.exists
    produces resource.updated
```

Expected:

```text
observe_resource
   |
   +-- exists == TRUE
   |       |
   |       v
   |   update_resource
   |
   +-- exists == FALSE
           |
           v
       create_resource
           |
           v
       update_resource
```

No CAMARA-specific knowledge is required.

---

# 40. Generic Multi-Intent Example

```text
Goal A:
produce fact A

Goal B:
requires fact A
produces fact B

Goal C:
independent
```

Expected:

```text
A -> B

C
```

No dependency should be introduced between C and A/B.

---

# 41. Impossible Plans

If:

```text
Goal:
fact Z
```

and:

```text
no operation produces Z
```

then:

```text
INFEASIBLE
```

Do not:

```text
invent operation
invent effect
invent dependency
ask LLM to repair graph
execute partial plan
```

Return a structured failure.

---

# 42. UNKNOWN vs UNSATISFIED vs INFEASIBLE

Keep these states distinct.

```text
UNKNOWN:
we do not know whether a condition holds.

UNSATISFIED:
we know it does not hold.

INFEASIBLE:
no valid operation sequence can establish it.

VALID:
the required condition is satisfied.
```

Example:

```text
qos.session.exists = UNKNOWN
```

means:

```text
find observation
```

not:

```text
create session
```

---

# 43. Planning Decisions

Expose deterministic planner reasoning:

```python
class PlanningDecision(BaseModel):
    operation_id: str
    applicable: bool
    satisfied: list[Condition]
    unsatisfied: list[Condition]
    unknown: list[Condition]
    reason: str
```

This is not natural-language reasoning.

It is an auditable symbolic decision trace.

---

# 44. Planning Trace

Optionally record:

```text
goal selected
condition evaluated
producer candidates
producer rejected
producer selected
dependency added
state transition
branch created
branch validated
cycle detected
goal satisfied
```

Example:

```text
[1] Goal qos.session.duration.updated
[2] Missing qos.session.id
[3] Producer candidates:
      retrieveSessionsByDevice
      getSession
[4] getSession rejected:
      requires qos.session.id
[5] retrieveSessionsByDevice selected
[6] Conditional branches constructed
[7] TRUE branch validated
[8] FALSE branch validated
[9] Complete plan valid
```

---

# 45. Explainability

The explanation layer should describe deterministic planner decisions.

Example:

```text
Goal:
extend active QoS session

Step 1:
retrieveSessionsByDevice

Reason:
qos.session.id is required by the goal-achieving operation.
The initial state does not contain qos.session.id.
The selected observation operation produces the missing fact.

Step 2:
extendQosSessionDuration

Reason:
qos.session.id is supplied by Step 1.
The requested duration is supplied by parameter extraction.
```

For a branch:

```text
Observation:
retrieveSessionsByDevice

TRUE:
qos.session.exists == TRUE
-> extendQosSessionDuration

FALSE:
qos.session.exists == FALSE
-> createSession
-> extendQosSessionDuration
```

The explanation should show:

```text
state_before
condition
decision
state_after
branch
validation result
```

---

# 46. Conditional Reporting

Before execution:

```text
Conditional Plan
----------------

Observation:
  retrieveSessionsByDevice

TRUE branch:
  extendQosSessionDuration
  VALIDATED

FALSE branch:
  createSession
  extendQosSessionDuration
  VALIDATED

Execution barrier:
  PASSED
```

After execution:

```text
Observed:
  qos.session.exists = TRUE

Selected:
  TRUE branch

Not executed:
  FALSE branch

Status:
  SUCCESS
```

The untaken branch is:

```text
pre-validated alternate branch
```

not:

```text
rejected
```

---

# 47. Runtime Fact Extraction

After execution:

```text
API response
     |
     v
produces_from
     |
     v
WorldState update
```

Example:

```text
response:
{
  "sessionId": "abc123"
}
```

becomes:

```text
qos.session.id = TRUE("abc123")
qos.session.exists = TRUE
```

If a list response is empty:

```text
[]
```

the observation should establish:

```text
qos.session.exists = FALSE
```

when the condition metadata defines that mapping.

A successful HTTP response is not automatically equivalent to a true resource-existence fact.

---

# 48. State Provenance

Runtime facts should record their source where possible.

Example:

```python
StateValue(
    truth=TRUE,
    value="abc123",
    source="retrieveSessionsByDevice.sessionId"
)
```

This supports:

- branch auditing;
- explainability;
- debugging;
- reproducibility.

---

# 49. Parameter Resolution Boundary

Parameter extraction remains outside PGPS.

Example:

```text
duration = 30 minutes
device.id = +491234567
activity = gaming
```

PGPS does not interpret those natural-language values.

It only reasons:

```text
this value is required
this value is already known
this value is supplied by a previous step
this value is unresolved
```

---

# 50. Prior-Step Binding

If:

```text
Step A:
produces qos.session.id

Step B:
requires qos.session.id
```

then:

```text
B.sessionId <- A.qos.session.id
```

is generated mechanically from the fact graph.

No per-operation table is required.

---

# 51. Recommended Core Models

Conceptually:

```python
class OperationKind(str, Enum):
    ACTION = "action"
    OBSERVATION = "observation"


class Condition(BaseModel):
    fact: str
    operator: Operator
    value: Any | None = None


class OperationConditions(BaseModel):
    operation_id: str
    kind: OperationKind

    requires: list[Condition] = []
    produces: list[str] = []
    updates: dict[str, Any] = {}
    invalidates: list[str] = []

    produces_from: dict[str, str] = {}
```

Plan:

```python
class PlanStep(BaseModel):
    step_id: str
    operation_id: str
    kind: OperationKind

    requires: list[Condition]
    produces: list[str]
    invalidates: list[str]
    updates: dict[str, Any]

    bindings: list[ParameterBinding]

    reason: str
    intent_ids: list[str]
```

---

# 52. Planner Contract

The target planner interface should conceptually be:

```python
class PGPSPlanner:

    def plan(
        self,
        *,
        goals: list[Goal],
        initial_state: WorldState,
        operations: list[OperationDefinition],
        conditions: ConditionModel,
        intent_graph: IntentGraph | None = None,
    ) -> PlanningResult:
        ...
```

Important:

```text
No LLM parameter.
No prompt.
No model.
No registry fallback.
```

The deterministic boundary should be obvious from the API.

---

# 53. Operation Catalog

PGPS should consume an operation catalog rather than depending directly on registry implementation details.

```python
class OperationDefinition(BaseModel):
    operation_id: str
    api_name: str
    method: str
    path: str
    parameters: list[ApiParameter]
    request_body: dict[str, Any] | None
    responses: list[dict[str, Any]]
```

The planner only needs normalized structured metadata.

---

# 54. Remove Legacy Registry Construction

The existing silent registry-only condition fallback is dangerous.

Avoid:

```python
PGPSPlanner(registry=registry)
```

when this can create an empty condition model.

Prefer:

```python
PGPSPlanner(
    condition_model=build_condition_model(...),
    operation_catalog=...
)
```

The planner should fail loudly when planning knowledge is absent.

This prevents the previously observed failure where real condition data existed but production wiring silently used an empty fallback.

---

# 55. Data Ownership

Recommended:

```text
data/
|
+-- camara_docs/
|     |
|     +-- OpenAPI-derived normalized knowledge
|
+-- conditions/
|     |
|     +-- PGPS planning semantics
|
+-- mock/
|     |
|     +-- runtime/mock responses
|
+-- chroma/
      |
      +-- retrieval index
```

Responsibilities:

```text
OpenAPI -> API semantics
RAG -> retrieval
conditions -> planning semantics
mock -> runtime data
Chroma -> retrieval infrastructure
```

---

# 56. Example Condition File

```json
{
  "retrieveSessionsByDevice": {
    "kind": "observation",

    "requires": [
      {
        "fact": "device.id",
        "operator": "exists"
      }
    ],

    "produces": [
      "qos.session.exists",
      "qos.session.id"
    ],

    "produces_from": {
      "qos.session.id": "sessionId",
      "qos.session.exists": "$exists"
    }
  },

  "createSession": {
    "kind": "action",

    "requires": [
      {
        "fact": "device.id",
        "operator": "exists"
      },
      {
        "fact": "qos.profile.id",
        "operator": "exists"
      }
    ],

    "produces": [
      "qos.session.exists",
      "qos.session.id",
      "qos.session.status"
    ],

    "updates": {
      "qos.session.exists": true,
      "qos.session.status": "ACTIVE"
    },

    "produces_from": {
      "qos.session.id": "sessionId"
    }
  },

  "extendQosSessionDuration": {
    "kind": "action",

    "requires": [
      {
        "fact": "qos.session.id",
        "operator": "exists"
      },
      {
        "fact": "qos.session.status",
        "operator": "eq",
        "value": "ACTIVE"
      }
    ],

    "goal_effects": [
      "qos.session.duration.updated"
    ]
  }
}
```

The exact schema can evolve, but the semantics must remain generic.

---

# 57. Deterministic Planning Example

Goal:

```text
extend an active QoS session by 30 minutes
```

Initial state:

```text
device.id = TRUE
duration = TRUE(30m)
qos.session.exists = UNKNOWN
qos.session.id = UNKNOWN
```

Conditions:

```text
retrieveSessionsByDevice:
    requires device.id
    produces qos.session.exists
    produces qos.session.id
    kind = observation

createSession:
    requires device.id
    requires qos.profile.id
    produces qos.session.exists
    produces qos.session.id
    kind = action

extendSession:
    requires qos.session.id
    requires qos.session.status == ACTIVE
```

PGPS determines:

```text
1. Goal requires extension.
2. Extension requires session.id.
3. session.id is unknown.
4. An observation can establish session.id.
5. The observation produces session existence information.
6. Both possible continuation plans are synthesized.
7. Both are validated.
```

Result:

```text
retrieveSessionsByDevice
       |
       +-- exists == TRUE
       |       |
       |       v
       |   extendSession
       |
       +-- exists == FALSE
               |
               v
           createSession
               |
               v
           extendSession
```

---

# 58. Multi-Intent Example

User:

```text
Find the best QoS profile for gaming, create a session using it,
and extend the session by 30 minutes.
```

Intent graph:

```text
I1:
Find best gaming QoS profile

I2:
Create QoS session

I3:
Extend session
```

Dependencies:

```text
I1 -> I2 -> I3
```

Facts:

```text
I1 produces qos.profile.id
I2 requires qos.profile.id
I2 produces qos.session.id
I3 requires qos.session.id
```

PGPS constructs:

```text
retrieveQoSProfiles
        |
        v
resolve selected profile
        |
        v
createSession
        |
        v
extendQosSessionDuration
```

The planner does not need to know what QoS means.

---

# 59. Multi-Intent With Independent Goals

User:

```text
Check my QoS profile and check my SIM-swap status.
```

If no dependency is declared:

```text
Intent A -> QoS plan

Intent B -> SIM-swap plan
```

Do not create:

```text
QoS -> SIM-swap
```

The planner should preserve independence.

---

# 60. Planning Failure Contract

Use a structured result:

```python
class PlanningResult(BaseModel):
    status: PlanStatus
    plan: Plan | None
    decisions: list[PlanningDecision]
    unresolved_goals: list[Condition]
    errors: list[PlanningError]
    warnings: list[str]
```

Possible statuses:

```text
VALID
INFEASIBLE
INVALID_INPUT
CYCLIC
AMBIGUOUS
MISSING_CONDITION_DATA
```

---

# 61. Safe Failure

When PGPS cannot build a valid plan:

```text
DO NOT:
- invent an operation
- invent an effect
- invent a dependency
- execute a partial plan
- ask an LLM to repair the graph
```

Instead:

```text
return structured failure
```

The workflow can then ask for clarification or stop.

---

# 62. Planning vs Runtime Failure

Keep separate.

### Planning failure

```text
missing producer
dependency cycle
invalid branch
missing static parameter
invalid payload
unreachable goal
```

### Runtime failure

```text
HTTP 409
HTTP 429
HTTP 500
timeout
network failure
backend state changed
authorization failure
```

A valid plan can still fail at runtime.

Deterministic planning guarantees static validity relative to supplied knowledge, not backend success.

---

# 63. Recommended Module Architecture

```text
src/camara_pgps/
|
+-- pgps/
|   |
|   +-- conditions.py
|   +-- conditions_loader.py
|   +-- world_state.py
|   +-- evaluator.py
|   +-- dependency_graph.py
|   +-- planner.py
|   +-- goal.py
|   +-- conditional_plan.py
|   +-- planning_decision.py
|   +-- parameter_facts.py
|   +-- costs.py
|   +-- explain.py
|
+-- models/
|   |
|   +-- operation.py
|   +-- plan.py
|   +-- execution.py
|   +-- intent.py
|
+-- retrieval/
|   |
|   +-- rag.py
|   +-- retriever.py
|
+-- agents/
|   |
|   +-- intent_agent.py
|   +-- api_selection_agent.py
|   +-- parameter_agent.py
|   +-- item_selection_agent.py
|
+-- validator/
|   |
|   +-- structural.py
|   +-- payload.py
|   +-- branch.py
|
+-- execution/
|   |
|   +-- executor.py
|   +-- mock_executor.py
|
+-- workflow/
|   |
|   +-- graph.py
|   +-- nodes.py
|   +-- report.py
```

---

# 64. Implementation Order

Do not rewrite everything at once.

Recommended sequence:

```text
1. Condition model
2. Predicate operators
3. WorldState
4. Condition evaluator
5. Condition loader
6. Goal model
7. Operation effect model
8. Dependency graph
9. Deterministic producer selection
10. Goal-oriented planner
11. ConditionalPlan / PlanBranch
12. Multi-intent integration
13. Prior-step bindings
14. Structural validation
15. Branch validation
16. Payload/schema validation integration
17. Executor state updates
18. Runtime branch selection
19. Explainability/reporting
20. Remove legacy registry fallbacks
```

---

# 65. Deterministic Input Normalization

Before planning:

```text
sort operation IDs
sort conditions
normalize fact names
normalize operators
normalize goal ordering
normalize intent ordering
normalize dependency ordering
```

Avoid dependence on:

```text
filesystem order
database result order
set iteration
dictionary insertion order
LLM output ordering
```

---

# 66. Reproducibility

A canonical input representation can support a hash:

```text
canonical_input
      |
      v
input_hash
      |
      v
PGPS
      |
      v
plan_hash
```

This enables later experiments demonstrating:

```text
same input -> same plan
```

---

# 67. Thesis Evaluation Strategy

The eventual evaluation should compare:

```text
Baseline:
RAG + LLM API sequence generation

Proposed:
RAG + LLM semantic components + deterministic PGPS + validator
```

Potential metrics:

```text
plan validity
dependency correctness
goal satisfaction
unnecessary operations
branch completeness
parameter completeness
schema validity
determinism/reproducibility
multi-intent success
```

The main research claim should focus on the planning layer, not LLM generation quality.

---

# 68. Thesis Contribution

The contribution should not be described as:

```text
"I generated a DAG."
```

or:

```text
"I used an LLM to generate API sequences."
```

A stronger statement is:

> **The thesis introduces PGPS, a generic deterministic planning layer for CAMARA API orchestration that converts grounded goals and API capabilities into dependency-aware plans using explicit preconditions, effects, observations, state transitions, and conditional branches, while keeping LLM-based semantic reasoning outside the authoritative planning process.**

A second contribution:

> **The framework pre-synthesizes and statically validates all reachable conditional branches before execution, allowing runtime observations to select only among already validated alternatives.**

A third:

> **The planner is operation- and API-agnostic because its planning behavior is derived from declarative conditions and effects rather than operation-specific procedural logic.**

---

# 69. Research Hypothesis

A suitable hypothesis is:

> **Explicit deterministic reasoning over preconditions, effects, dependencies, and state transitions produces more structurally valid, complete, and explainable CAMARA API orchestration plans than semantic retrieval combined with unconstrained LLM-generated call ordering.**

---

# 70. What PGPS Does Not Do

PGPS does not:

```text
understand natural language
retrieve documents
generate embeddings
choose wording
invent API capabilities
execute HTTP requests
interpret arbitrary natural-language backend semantics
select list items based on language
```

Its job is:

```text
given structured knowledge,
reason deterministically about what must happen.
```

---

# 71. Future Extension: Online Replanning

Full online replanning can remain future work.

Possible future architecture:

```text
observe
  |
  v
replan
  |
  v
observe
  |
  v
replan
```

The current thesis deliberately prioritizes:

```text
pre-plan
pre-validate
execute
observe
select prevalidated branch
```

because this provides a clear execution barrier for side-effecting telecom workflows.

---

# 72. Future Extension: Recovery Planning

Future work may introduce:

```text
runtime failure
      |
      v
recovery planner
```

but recovery planning should remain deterministic and validated before execution.

It must not silently turn the executor into a second unconstrained planner.

---

# 73. Future Extension: Parallel Execution

Once dependencies are formally represented, independent operations can be detected.

Example:

```text
A       C
|       |
B       D
```

If there are no dependency or shared-resource conflicts, A/C may eventually execute in parallel.

This is an optimization and not required for the core thesis.

---

# 74. Recommended Thesis Architecture Statement

> The proposed system separates semantic interpretation from authoritative planning. Natural-language processing, retrieval, capability selection, and parameter extraction are handled by LLM-based components that produce grounded structured inputs. These inputs are passed to a deterministic Precondition-Guided Plan Synthesis layer. PGPS reasons over explicit goals, operation preconditions, effects, observations, dependencies, and world-state transitions to construct a complete plan. When runtime state is uncertain, PGPS synthesizes all relevant conditional branches in advance and validates them before execution. During execution, backend observations update the world state and select only among branches that have already passed validation. Consequently, no branch is generated or semantically repaired after a side-effecting operation has started.

---

# 75. Final Thesis Position

The strongest conceptual separation is:

```text
LLM
=
semantic intelligence

RAG
=
capability grounding

PGPS
=
deterministic planning intelligence

Validator
=
deterministic correctness gate

Executor
=
runtime state interpreter
```

The thesis does not attempt to make an LLM perfectly plan telecom APIs.

Instead:

> **LLMs understand what the user wants and ground the available capabilities; deterministic PGPS decides what must happen, in what dependency order, under which conditions, and whether every possible execution path is valid before anything reaches the live telecom backend.**

The central research property is:

> **No branch is invented after execution begins.**

The central genericity property is:

> **A new API should be supported by declarative operation and condition data, not by changing PGPS planner code.**

The central determinism property is:

> **Given identical normalized goals, state, operations, conditions, intent dependencies, and planning policy, PGPS produces the same plan and validation result.**

