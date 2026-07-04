# Agent layer — bounded autonomy over the tiered pipeline

> Status: **scaffolding**. The loop, tools, grounding gate, and offline policy are
> implemented and tested; the live LLM policy is implemented but not measured at
> scale. This document describes what the agent *is* and, just as importantly,
> what it is deliberately *not allowed* to do.

## What this is

`metaextract.agent` is an autonomous **tool-use loop** that assembles
treatment-vs-control records from a paper that has already been screened and
located. Instead of a human clicking each located region in the cockpit, a
*planner* decides the next action, the loop runs the corresponding tool, and the
result feeds the next decision — a real perceive → act loop.

## What this is **not**

It is not a new, higher-trust way to read numbers. The project's core finding is
that a silent transcription error is the most expensive failure in a
meta-analysis, so AI is **never** allowed to commit a number the source does not
support. The agent inherits that rule unchanged: every value it commits passes the
same deterministic grounding gate (`extract.ground_verify`) the cockpit uses, and
a value that cannot be grounded is flagged for a human, never silently written.

The agent can choose *what to do next*. It cannot choose to *trust a number the
source does not support*. That is the whole design.

## Architecture

```mermaid
flowchart TB
    subgraph policy["Planner (the policy — swappable)"]
        RP["ReplayPlanner<br/>deterministic, offline, no API"]
        GP["GeminiPlanner<br/>live LLM function-calling"]
    end

    policy -->|next action| LOOP["Agent loop<br/>(dispatch + step budget + trace)"]
    LOOP -->|tool call| TOOLS

    subgraph TOOLS["Tools — thin wrappers over existing, tested stages"]
        T1["list_regions / read_block"]
        T2["infer_columns / parse_table<br/>(tabular.py geometry)"]
        T3["find_sample_sizes<br/>(sampling.py regex)"]
        T4["propose_values / extract_values"]
        T5["commit_record"]
        T6["flag_for_review / finish"]
    end

    T4 --> GATE
    T5 --> GATE
    GATE{"Grounding gate<br/>extract.ground_verify<br/>value ⊆ span ⊆ block"}
    GATE -->|pass| REC["VerifiedRecord<br/>status = verified"]
    GATE -->|fail| FLAG["needs_review / flagged"]

    LOOP -->|result| policy
    LOOP --> TRACE["AgentResult<br/>records + flags + full trace"]
```

The loop is intentionally dumb — it has no judgement, only a step budget. The
bounded-autonomy guarantee lives in the **tools**, not the loop: `commit_record`
sets a record's status from `gate.record_status`, never from the model.

## The tools

| Tool | Wraps | Purpose |
| --- | --- | --- |
| `list_regions` | `LocateOutput` | the already-located target regions |
| `read_block` | `SourceDoc` | one block's text + numeric tokens |
| `infer_columns` | `tabular.column_count` | how many value columns a table block has (label-free) |
| `parse_table` | `tabular.parse_block` + `make_pairings` | rows × columns → control×treatment candidates |
| `find_sample_sizes` | `sampling.find_sample_size_candidates` | deterministic replicate-count (n) candidates |
| `ground_check` | `extract.ground_verify` | probe the gate for one value **without** committing it |
| `propose_values` | `extract.apply_grounding` | submit values; each is gated before storage |
| `extract_values` | `extract.extract_region` | **live** Tier-3: model reads the block, then the gate runs (needs API) |
| `commit_record` | `gate.assemble_record` | materialize a `VerifiedRecord`; status comes from the gate |
| `flag_for_review` | — | hand a region to a human instead of committing |
| `finish` | — | end the run |

## The two planners

Both drive the **identical** tools and the identical gate, so the autonomy is
bounded the same way regardless of who is steering.

- **`ReplayPlanner`** — a deterministic policy that reacts to real tool output
  (column count, parsed candidates). Written as a generator so it reads like a
  script while still being a genuine perceive→act loop. Needs no API key, so it
  runs the demo and the test suite in CI. It uses generic column labels
  (`col1`/`col2`) and is a *baseline* policy, not a claim of correct pairing.
- **`GeminiPlanner`** — the live LLM policy via Gemini function-calling. It can
  supply real column labels and use `extract_values`. Implemented for the
  `--online` path; not exercised in CI.

## What the gate catches — and what it does not

The grounding gate catches the failures the model's own confidence cannot:

- a **hallucinated number** (not present in the cited block) → rejected
- a value **spliced from a different table** (its evidence span isn't in the
  cited block) → rejected
- a non-numeric value painted as a measurement → rejected

It does **not** catch a number that is real but assigned to the **wrong row** —
e.g. a block that merges `pH + Moisture + Total N` into one region, where the
first row's (real) values could be handed to all three variables. The number
grounds; the *identity* is wrong. This is exactly the row-identity matching the
project deliberately leaves to a human (Architecture v3.1). So the agent does not
guess it: when `parse_table` returns more than one row, the policy
`flag_for_review`s the region (`multi_row_block_needs_row_assignment`) instead of
committing. The demo below shows this happening on a real paper.

## Running it

Offline (no API key), over a cached paper:

```bash
python scripts/agent_demo.py P1
# or
metaextract agent P1 --report data/outputs/agent_run_P1.json
```

Live LLM policy (needs a Gemini key; same tools, same gate):

```bash
metaextract agent P1 --online
```

On cached P1 the offline run commits the clean single-row tables as `verified`
records (their values matching the hand-extracted ground truth) and flags the
merged `pH/Moisture/TN` block for human row-assignment — a faithful demonstration
of the gate accepting grounded values, rejecting a fabricated probe, and the
policy deferring the matching it should not automate.

## Relationship to the rest of the system

The agent is an **orchestrator**, not a new tier. It sits on top of:

- **Tier 1/2** (`locate.py`) — screening + locating, run once and cached.
- **Tier 3** (`extract.py`) — on-demand AI extraction behind the grounding gate;
  the agent calls it via `extract_values` and never bypasses the gate.
- **Deterministic helpers** (`tabular.py`, `sampling.py`) — geometry parsing and
  the n finder, the same code the cockpit uses.

Every record the agent emits is a normal `VerifiedRecord` with per-slot citations,
so its output is as auditable and jumpable as a human's — plus a full tool-call
trace of how each value was accepted or deferred.
