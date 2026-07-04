# MetaExtract — Architecture v3.1

A **human-in-the-loop verification cockpit** for meta-analysis data extraction.

v3 keeps v2's mechanisms (addressable source, evidence-carrying values,
deterministic grounding, declarative validation) but **changes the defaults and
the product form**:

- **Product form** — a local-first, single-screen *verification cockpit* for
  personal / lab use, built around one thing: *click a record → instantly see
  its source span in the PDF*. (See [ARCHITECTURE_v2.md](ARCHITECTURE_v2.md) for
  the framework rationale this builds on.)
- **Default autonomy** — v2 still extracted every number and then asked a human
  to review. v3 makes **AI screen + locate by default, and only extract numbers
  on demand**. Autonomy is matched to *error cost*.

> **Status note.** v2 was a design target, not the shipped code. The current
> code ([schema.py](../src/metaextract/schema.py),
> [extractor.py](../src/metaextract/extractor.py)) does flat one-shot extraction
> (plain `float` fields, coarse `Provenance` like `"Table 2"`, no evidence span,
> no grounding). v3 is the blueprint for moving from that to the cockpit. Each
> section below marks **[exists]**, **[upgrade]**, or **[new]**.

---

## 0. Changes from v3.0 (after review)

v3.0 was reviewed; five findings were accepted. The keystone (C3) is a data-model
correction — the other changes hang off it.

- **C1 — jump-to-evidence needs more than "required fields".** `Citation` now
  carries `page` + `bbox` + an optional word-level span, all from **deterministic
  ingest, not the LLM**. The locator LLM **selects a pre-addressed `block_id`**;
  it never describes a location in prose. Marking a string field `required` would
  not fix this — the producer (Gemini saying `"Table 2"`) can't fill a bbox.
  See §3, §4.
  - *Implemented (M0a) & corrected by evidence.* `find_tables()` returns **zero**
    tables on the test paper (borderless journal tables have no ruling lines) and
    errors on the text strategy, so a deterministic **cell-id grid is not
    obtainable**. Replaced with **word-level addressing**: every token keeps an
    exact bbox, so any value (e.g. `0.302***a`) is highlighted precisely. Proven
    end-to-end on the real PDF. See §4, §8.
- **C2 — human-filled values are no longer "trusted by definition".** They become
  `human_verified` with citation + timestamp + reviewer action, and a **passive,
  non-blocking** deterministic typo-check when the value is text-matchable. The
  rationale is provenance + a free typo-catch (the audit trail *is* the product),
  not "humans are as untrustworthy as the LLM". See §5.
- **C3 — `LocatedRecord` presupposed pairing; it contradicted Tier 2.** Replaced
  by a three-layer model: **`LocatedRegion` → `ExtractionSlot` → `VerifiedRecord`**.
  A record only materializes *after* a human completes pairing. This also closes
  the moderator-binding trap at the contract level. See §4.
- **C4 — moderator manual-binding was undesigned; `flatten.py` fakes bindings.**
  Added `moderator_binding_status`; unbound moderators export to a **separate
  paper-level table**, never denormalized onto rows. The old
  [flatten.py](../src/metaextract/flatten.py) row-repeat logic is an anti-pattern
  here and is **not carried forward**. See §4, §7.
- **C5 — M0 had no evaluation protocol.** M0 now defines metrics, prioritizing
  *time per verified record* (vs a manual baseline) and *location recall*, plus a
  `TableS1.xlsx → truth.csv` + source-map conversion. One paper is a qualitative
  smoke test, not a benchmark. See §8.

---

## 1. Why the defaults changed: error cost

AI does two jobs here with error costs an order of magnitude apart:

| Job | Wrong-answer cost |
|-----|-------------------|
| **Locate / classify** ("the N2O data is in Table 3") | Cheap — a human spots it instantly ("there is no Table 3"). |
| **Transcribe a number** (read `4.2`, pair SD to the right group) | Expensive **and silent** — a wrong `4.2`/`2.4` corrupts the effect size and nobody notices. |

So the design rule is: **let autonomy track error cost.** AI is trusted to
locate and classify; it is *not* trusted to silently transcribe. The product's
job is to make the human's confirm-a-number step take ~2 seconds.

---

## 2. Three tiers of AI autonomy

```
Tier 1  SCREEN + LIST        full-auto    高信任   错了便宜
        纳入/排除 · 出现的目标变量 · 出现的 moderator

Tier 2  LOCATE (don't extract)   default   中信任   产出"位置图"，不产出数字
        每个目标 → 一条 LocatedRegion {变量, 在哪个 block, 候选结构}

Tier 3  EXTRACT on demand        opt-in    低信任   必须带 span + 过 ground-verify
        人点某个已定位 slot 才触发；产出对不上原文就 flag 红
```

- **Tier 1** output is a per-paper *screening worksheet*: include/exclude, which
  target variables are present, which moderators are present. Reliable, and
  errors are cheap to catch.
- **Tier 2** output is a *location map* of `LocatedRegion`s. Its unit is **a
  region citation, not a number, and not a paired record** (§4). This is the new
  first-class artifact.
- **Tier 3** is the only path that produces AI numbers, and every such number
  must round-trip through deterministic grounding before a human ever sees it as
  a suggestion.

### The trap this avoids: moderator → record binding

"Simple Host/moderator variables can be auto-extracted" is true **at the paper
level** ("list the soil types mentioned") but false **at the record level**: a
paper with 3 sites / different soils requires each N2O number to attach to the
*correct* site's moderators. That binding **is** the complex matching we are
avoiding.

Rule: **moderator→record binding stays on the human side.** v3.1 enforces this in
the data contract: moderators live as paper-level facts; a `VerifiedRecord`
(§4) only exists after a human assembles it, and it carries an explicit
`moderator_binding_status`. There is no code path that auto-binds a moderator to
a row.

---

## 3. The product: a verification cockpit  [new]

One local screen. Streamlit first (fast, same Python ecosystem as the existing
package; the pipeline is decoupled by stage contracts so the frontend can be
swapped later without touching the backend).

```
┌──────────────────────────────────────────────────────────────┐
│  TOP  — Screening summary (Tier 1)                             │
│  include/exclude · target variables present · moderators       │
├──────────────────────────────┬─────────────────────────────────┤
│  LEFT — PDF page viewer       │  RIGHT — slots for this paper    │
│  evidence span highlighted    │  one row = one ExtractionSlot    │
│  (click a slot → jump here)   │  value cell = "located · empty"  │
│                               │  actions: fill / pair / reject   │
│                               │  status: located · verified·flag │
└──────────────────────────────┴─────────────────────────────────┘
```

**Make-or-break feature: jump-to-evidence.** The entire value of the product is
*evidence sitting next to the number, one click away*. If clicking a slot does
not immediately show the exact spot in the PDF, the value evaporates.

**Its hard dependency (C1).** Click-to-jump-and-highlight only works if every
addressable block already has `page` + `bbox` (and table cells have a `cell_id`)
**from deterministic ingest**. Therefore:

- ingest (PyMuPDF / table parser) assigns `block_id ↔ (page, bbox)` up front;
- the Tier 2 locator LLM **chooses an existing `block_id`**, it does not invent a
  prose location;
- bbox is never produced by the LLM.

This is why **PDF ingest with bbox is the first thing M0 builds**, before the
cockpit — without it the core feature can only jump to a page, not highlight.

### Per-paper workflow

1. Drop a PDF → ingest to `SourceDoc` (blocks with page+bbox) → run Tier 1
   (screening) + Tier 2 (locate) once; cache it.
2. Cockpit opens with the worksheet + `ExtractionSlot`s; **all numbers start
   empty**, each tagged with the block it lives in.
3. Human goes slot by slot: click → PDF jumps & highlights → fill / confirm the
   number, and **pair** treatment↔control → accept. Pairing promotes slots into a
   `VerifiedRecord`.
4. On demand, trigger Tier 3 AI extraction for a specific slot; the suggestion is
   shown only after passing grounding, red-flagged if it fails.
5. Figures → "open in digitizer" handoff (§6), never AI pixel-reading.
6. Export `meta_ready.csv` + `moderators.csv` + `evidence_report.html`.

---

## 4. Data model: three layers  [new — replaces v3.0 `LocatedRecord`]

The keystone fix. Tier 2 can usually only say *"Table 2 contains the target
variable"* — it often cannot yet say which treatment/control/mean/sd/n line up.
Forcing a paired record at that stage (v3.0's `LocatedRecord` with required
`mean_t`/`mean_c`) re-introduces the exact matching problem we deferred to humans.
So the pipeline has **three distinct artifacts**; a record only exists after human
assembly.

```python
# --- shared: a precise, verifiable citation (C1) ---
# Implemented as metaextract.sourcedoc.BBox + a word-level resolver
# (metaextract.highlight.locate_value), NOT a table cell grid — see C1 note.
class Citation(BaseModel):
    block_id: str                 # stable, chosen from the ingested SourceDoc
    page: int                     # required — from deterministic ingest
    bbox: tuple[float, float, float, float]   # required — from ingest, not LLM
    kind: Literal["paragraph", "figure"]
    evidence_text: Optional[str] = None # exact span; resolved to word bbox(es)
                                        # by highlight.locate_value for the cockpit

# --- Tier 2 output: a located region, NO pairing, NO numbers ---
class LocatedRegion(BaseModel):
    region_id: str
    paper_id: str
    variable_name: str            # the target variable believed present here
    citation: Citation            # where it lives (jumpable)
    candidate_structure: Optional[str] = None  # free-text hint: "rows=treatments,
                                               # cols=mean/sd/n", NOT a commitment
    ambiguous: bool = False       # AI unsure whether the variable is really here

# --- worksheet unit: an empty cell waiting for a value ---
class ExtractionSlot(BaseModel):
    slot_id: str
    region_id: str                # which LocatedRegion it belongs to
    role: Literal["mean_t","sd_t","n_t","mean_c","sd_c","n_c"]
    citation: Optional[Citation] = None        # narrows to the cell, once known
    value: Optional[float | int] = None        # None = located, not yet filled
    unit: Optional[str] = None
    source: Literal["empty","human","ai_extracted"] = "empty"
    grounding_verified: bool = False           # set by §5, never by the model
    flags: list[str] = Field(default_factory=list)

# --- only after a human pairs slots into a treatment/control point ---
class VerifiedRecord(BaseModel):
    record_id: str
    paper_id: str
    source_block_ids: list[str]   # record-level evidence (binds the slots)
    variable_name: str
    treatment_group: str
    control_group: str
    slots: dict[str, ExtractionSlot]           # the six filled+verified slots
    moderator_binding_status: Literal["unbound","bound"] = "unbound"
    bound_moderators: Optional["Moderators"] = None  # set only when bound
    status: Literal["needs_review","verified","invalid"] = "needs_review"
    flags: list[str] = Field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None          # ISO timestamp
```

Notes:

- **`StudyInfo` / `Moderators`** from the current schema stay as paper-level
  facts for Tier 1 output. They are **not** auto-bound to records (§2). A
  `VerifiedRecord` only carries moderators once `moderator_binding_status ==
  "bound"`, which a human sets.
- **Implementation can be lean (M0).** Build `LocatedRegion` + `ExtractionSlot`
  as the working artifacts and use `VerifiedRecord` purely as the **export
  gate** — don't over-engineer three deep Pydantic hierarchies for a personal
  tool. The *separation* is what matters, not ceremony.

---

## 5. Trust: grounding for AI, provenance + typo-check for humans

### 5.1 Tier 3 (AI) — deterministic grounding gate  [new, no LLM]
For any slot whose `source == "ai_extracted"`:

- the printed `value` must appear (tolerant numeric match) in
  `citation.evidence_text`;
- `evidence_text` must appear (normalized) inside the cited block's text/cell;
- `citation.block_id` must be in the record's `source_block_ids`.

Pass → `grounding_verified = True`. Any failure → flag `evidence_unverified`,
force `needs_review`, show **red** in the cockpit. Cheap, deterministic, catches
the dominant hallucination/splice failures LLM confidence cannot.

### 5.2 Humans — provenance + passive typo-check (C2)
Human input is **not** "trusted by definition". The human is the trust authority
(there is no higher oracle to check against), but:

- every human value carries a `citation`, `reviewed_by`, `reviewed_at`, and the
  action taken — **provenance is the product's whole selling point**;
- when the value is text-matchable to its cited block, run the *same*
  deterministic check as a **passive, non-blocking typo-catch**: a mismatch is
  highlighted, it does **not** block acceptance. (Blocking every cell would turn
  the cockpit into a data-entry jail and kill the speed that is its reason to
  exist.)

### 5.3 Figure-digitized values
Stored as `source == "human"` with extra evidence: the **axis calibration**,
the **clicked point coordinates**, and a **screenshot** of the digitized region.
This is what makes a figure number auditable later.

---

## 6. Figures: locate, then hand off  [new]

AI reading numbers off a plot is the least reliable step in the pipeline. AI's
only job on figures is **Tier 2**: identify the figure block, which series /
axis is the target, and the candidate structure. The number itself comes from a
digitizer (WebPlotDigitizer-style) where a **human calibrates the axes**. The
cockpit links out to that step and ingests the result as a §5.3 figure-digitized
value. AI never reads pixels as values.

---

## 7. Pipeline (v3.1)

```
PDF
  │  ingest                         [upgrade] → SourceDoc (blocks w/ page+bbox)
  ▼
SourceDoc
  │  screen        (Tier 1, LLM)    [new]     → screening worksheet + moderators
  │  locate        (Tier 2, LLM)    [new]     → list[LocatedRegion]  (no numbers)
  ▼
location map + worksheet (ExtractionSlots, empty)
  │  COCKPIT       (human-driven)   [new]
  │    ├─ fill / confirm slot                  → ExtractionSlot.value (human)
  │    ├─ extract (Tier 3, on demand, LLM)     → ExtractionSlot.value (ai_extracted)
  │    │     └─ ground-verify  [new, det.]     → grounding_verified
  │    ├─ pair treatment↔control               → VerifiedRecord (status=needs_review)
  │    └─ bind moderators (explicit)           → moderator_binding_status=bound
  ▼
list[VerifiedRecord]
  │  validate                       [upgrade from validator.py]
  │  normalize     (UnitRegistry)   [new, optional for MVP]
  ▼
export                              [new — NOT flatten.py's row-repeat]
  → meta_ready.csv   (only bound records; one row per VerifiedRecord)
  → moderators.csv   (paper-level + unbound moderators, kept separate)
  → evidence_report.html · review_log.csv
```

Reuse map: `schema.py` → three-layer model above; `extractor.py` → split into
`screen` + `locate` (+ optional `extract`), not one-shot; `validator.py` →
validate stage; `pipeline.py` → caching of screen/locate results; `evaluate.py`
→ extend per §8. **`flatten.py` is deprecated** for this design — its
study-level-moderator-repeated-per-row logic ([flatten.py:23](../src/metaextract/flatten.py))
manufactures false bindings in multi-site / multi-soil / multi-depth papers
(C4). Export is rewritten to keep moderators separate until bound.

---

## 8. Build order (MVP first, personal/lab)

- **M0a — PDF ingest with bbox (build first, C1).** ✅ *Done.* PyMuPDF →
  `SourceDoc` ([sourcedoc.py](../src/metaextract/sourcedoc.py),
  [ingest.py](../src/metaextract/ingest.py)) where every block has `block_id`,
  `page`, `bbox`, and **every word keeps an exact bbox**. A deterministic
  resolver ([highlight.py](../src/metaextract/highlight.py),
  `locate_value`) turns an evidence string into the highlight box, tolerant of
  significance marks and unicode minus. `find_tables()` is **not used** (returns
  nothing on borderless tables). Verified end-to-end on the test PDF; covered by
  [test_highlight.py](../tests/test_highlight.py). Without this the cockpit could
  only jump to a page, not highlight.
- **M0b — locate + cockpit on ONE paper.** Use
  [1-s2.0-S0167880914000905-main.pdf](../../1-s2.0-S0167880914000905-main.pdf)
  with `TableS1.xlsx` as ground truth.
  - Tier 1 screen + Tier 2 locate (LLM picks `block_id`s); **no auto-extract.**
    🟡 *Scaffolding done* ([locate.py](../src/metaextract/locate.py),
    [records.py](../src/metaextract/records.py) three-layer model;
    [test_locate.py](../tests/test_locate.py)). `build_payload` +
    `resolve_regions` (the C1 guard: invented `block_id`s are dropped+reported,
    never given coordinates) are deterministic and tested. The Gemini call is a
    thin wrapper, **pending an API key** for a live run.
  - Streamlit cockpit: PDF + worksheet, **click-to-jump-and-highlight**, human
    fill + pair, export `meta_ready.csv`. ✅ *Done* ([app/cockpit.py](../app/cockpit.py),
    [render.py](../src/metaextract/render.py),
    [cockpit_cache.py](../src/metaextract/cockpit_cache.py)). Loads offline from a
    JSON cache (`scripts/prepare_paper.py`); selecting a region jumps+outlines the
    block, typed values light up via word-level highlight.
  - **Multi-pairing assist** ([tabular.py](../src/metaextract/tabular.py),
    [test_tabular.py](../tests/test_tabular.py)): geometry parses a table block
    into rows×columns and reuses the control column across treatments (one click →
    NF→AF/SL/FL pre-filled, each value highlighted, human accepts). Parsing is
    **row-aware**: ingest sometimes merges several short table rows into one block
    (P1 p4.b14 = pH+Moisture+Total-N); columns come from value-x clustering, rows
    from the left-most name-column anchors (mean/sd y-gaps are too close to split
    on). A merged block surfaces a row-picker so the human chooses — never a silent
    mis-aligned value. *(Found+fixed via end-to-end: pre-fix, pH/Moisture/TN all
    got pH's numbers.)*
  - **Moderator binding** (v3.1 §2, C4). `locate` also extracts paper/site-level
    moderator *values* (site, MAP/MAT, soil, depth, years, climate) each with a
    citation; the cockpit shows them in an editable panel (AI suggests, human
    confirms) and binds the confirmed set to every exported record with
    `moderator_binding_status`. Old/New use stay the pairing, not moderators — so
    binding is human, never auto-attached. Export now yields target-shaped rows:
    moderators + control→treatment + per-variable Xc/Sc/Xe/Se + provenance.

- **M0 — evaluation protocol (C5).** One paper is a **qualitative smoke test +
  one timing anecdote**, not a benchmark — say so. Required deliverables:
  - a `TableS1.xlsx → truth.csv` + **source-map** converter (which table/figure
    each truth value came from), so location can be scored at all;
  - **primary metrics** (the M0 hypothesis): *time per verified record* vs a
    timed manual pass on the same paper; *location recall* (did Tier 2 point at
    every target variable — misses are the dangerous failure);
  - **secondary**: location precision, human correction rate, evidence
    completeness, ambiguous-region rate, missed-target count.
  - 🟢 *First numbers* ([scripts/eval_against_gt.py](../scripts/eval_against_gt.py)):
    P1 (table paper) — the cockpit's geometry auto-pairing **reproduces hand
    extraction exactly**: record recall 54/54 = 100%, and **all six numeric fields
    (mean_c / sd_c / mean_t / sd_t / n_c / n_t) 100%** within 2% (18 vars × 3
    treatments). Match is value-anchored (predicted mean_t ≈ GT Xe) with strict
    variable alignment + nearest-treatment. Figure papers (P2) go through the
    digitizer, not auto-pairing. One clean paper is a smoke test, not a benchmark.
  - **n-value location** ([sampling.py](../src/metaextract/sampling.py),
    [test_sampling.py](../tests/test_sampling.py)). n (Nc/Ne) isn't in the data
    table — it's one statement in methods/footnote, and the **LLM proved unreliable**
    at it (on P1's "84 samples (7 sites × 4 land use types × 3 replicates)" it
    picked 5, then 7, never the statistical 3). A **deterministic regex** finds the
    "3 replicates" phrase reliably; the cockpit surfaces the candidate(s) with the
    source phrase inline and the human confirms → Nc/Ne default to it. Clean case
    of "locate deterministically, human decides" beating the LLM.
  - **Figure-paper path** ([render.py](../src/metaextract/render.py) `render_region`).
    P2 is ~all figures (bar charts). For a figure region the cockpit shows a zoomed
    **crop** of the chart + its **caption**, and a digitized-value entry form
    (control/treatment + Xc/Sc/Xe/Se + n) tagged `source=figure_digitized` with a
    **digitizer-ref** field (file / screenshot / axis calibration) — the §6 handoff.
    AI located the figure + which variables it holds (from the caption); the human
    digitizes the numbers. Both paper classes are now actionable end-to-end.
  - **Corpus benchmark + HITL-lean pivot** (P1–P12, [scripts/benchmark.py](../scripts/benchmark.py),
    [bench_values.py](../scripts/bench_values.py)). Location recall ~79% (89%
    excluding 3 anomalies); type mix 7 table / 1 figure / 4 mixed. But value
    auto-pairing only generalized to P1 (100%) — the corpus has **diverse table
    structures**: normal (variable-per-row), **transposed** (P9: variables in
    columns, with nested depth sub-columns), and locate-points-at-caption (P12/P10).
    **Auto-classifying structure proved unreliable** (two heuristics contradicted:
    7 vs 0 transposed). Decision: **auto-pairing is a *fast path* for simple normal
    tables only; the general, reliable core is locate → render the table → manual
    entry with provenance**. Manual entry now highlights a typed value **anywhere on
    the page** (`_locate_on_page`), so verification works on transposed/nested
    tables too (proven on P9). Lesson reinforced: don't make AI (or a fragile
    heuristic) guess table structure — let the human decide, give them fast
    verification. 3 anomalies flagged: P3/P4 likely wrong PDFs (P3.pdf is a PAH
    paper), P8 GT rows lack readable values.

- **M1 — Tier 3 on-demand extraction** behind grounding; red-flag failures;
  review log feeds examples. 🟡 *Scaffolding done* ([extract.py](../src/metaextract/extract.py),
  [test_extract.py](../tests/test_extract.py)). The deterministic grounding gate
  (§5.1) is built and tested: `ground_verify` checks the printed value appears in
  the model's own quoted span (symbol/unicode-minus tolerant, shared with
  [highlight.py](../src/metaextract/highlight.py) `numeric_in_text`) and that the
  span appears in the cited block; `apply_grounding` turns the raw response into
  `ExtractionSlot`s, setting `grounding_verified` and flagging `evidence_unverified`
  on failure — the Tier-3 analogue of locate's C1 guard, testable without the API.
  The cockpit wires a per-region **"🤖 AI extract"** button: grounded values pre-fill
  the manual-entry form (and light up via the existing word-level highlight),
  ungrounded values show **red** and are never auto-filled; each run appends an
  `ai_extract` row to the review log (corrections are then captured by the existing
  add/delete log — the example feed). `extract_region` is a thin Gemini wrapper,
  **pending an API key** for a live run (same status as `locate`).
- **M2 — second+ papers + validate/normalize**; lnRR effect-size columns on
  export; first *statistically* meaningful metrics here, not in M0.
- **M3 — figure digitizer handoff** (§5.3, §6); batch over a folder.
- **Later** — TaskPack generality (v2 §1) only if a second domain actually shows
  up. For personal/lab use, one well-tuned soil/agri pack beats premature
  generality.

---

## 9. Explicitly out of scope (v3)

- Full meta-analysis platform — effect-size modeling / forest plots stay in
  R/metafor. We produce a trustworthy record table, nothing more.
- Fully automatic extraction (rejected: silent numeric error is catastrophic).
- Cloud SaaS / multi-user / accounts — local-first, single user.
- AI reading numbers off figures (§6).
- Auto-binding moderators to records (§2, enforced in the data model).

---

## 10. One-line positioning

The product is a **verification cockpit**, not an extractor: AI screens and
locates by default, a record only exists after a human pairs and verifies it,
evidence is one click away, and any AI-suggested number is grounded before it's
shown — so the bottleneck we optimize is *human verification*, which is where the
real value and trust live.
