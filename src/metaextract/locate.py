"""Tier 1 screen + Tier 2 locate (Architecture v3.1 §2, §7).

The locator is handed the *already-addressed* blocks of a :class:`SourceDoc` and
asked to do two cheap, low-error-cost jobs:

1. **Screen** the paper — include/exclude, which target variables and moderators
   are present (paper level only; never bound to a record, v3.1 §2).
2. **Locate** — for each target variable, name the ``block_id`` that contains it.
   It returns NO numbers and does NO treatment/control pairing.

The critical-path invariant (C1): the model may only *choose a block_id that
already exists*; all geometry (page, bbox) is attached afterwards from the
SourceDoc by :func:`resolve_regions`. A block_id the model invents is dropped and
reported, never given fabricated coordinates. ``build_payload`` and
``resolve_regions`` are pure and unit-tested without the API; :func:`locate` is a
thin Gemini wrapper around them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .families import DEFAULT_FAMILY, RESERVED_RECORD_KEYS, get_family
from .records import Citation, ExtractedModerator, LocatedRegion, SampleSize, ScreeningResult
from .sourcedoc import SourceDoc

# How much of each block's text the model sees. Enough to recognize a variable;
# the model never needs to parse the grid (that is human/Tier-3 work).
_BLOCK_TEXT_CHARS = 320


class TargetVar(BaseModel):
    """One target metric the researcher wants located.

    ``name`` is the canonical name (used for the located region, export columns,
    and cross-paper alignment). ``aliases`` are the synonyms papers actually use
    (e.g. SOC / TOC / "organic carbon") — the locator normalizes any of them back
    to ``name``. A bare string is accepted anywhere a ``TargetVar`` is expected
    and becomes ``TargetVar(name=...)`` with no aliases.
    """

    name: str
    label: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    unit_hint: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target variable name must not be blank")
        return value

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for alias in value:
            alias = alias.strip()
            if not alias:
                continue
            key = alias.casefold()
            if key not in seen:
                seen.add(key)
                out.append(alias)
        return out

    def as_prompt_line(self) -> str:
        """Canonical name + synonyms + optional unit, as one LLM-prompt bullet.

        Shared by the locator and the extractor so the alias/unit wording (and
        the "report under the canonical name" convention) lives in one place.
        """
        line = self.name
        if self.aliases:
            line += f" (also written as: {', '.join(self.aliases)})"
        if self.unit_hint:
            line += f" [typical unit: {self.unit_hint}]"
        return line


class TaskSpec(BaseModel):
    """The domain knob: which metrics to locate and which moderators to note.

    Edit this as a YAML "task pack" (see ``data/taskpacks/*.yaml`` and
    :meth:`from_yaml`) — researchers swap target metrics by editing that file, no
    code change. ``target_variables`` accepts either rich ``TargetVar`` entries or
    bare strings (kept for backward compatibility with the GT-derived specs).
    """

    domain: str
    target_variables: list[TargetVar]
    moderators: list[str] = Field(default_factory=list)
    # Which meta-analysis design this pack collects (see families.py). Decides the
    # per-record field set / cockpit form. Defaults to the only family validated on
    # real papers; others are experimental.
    analysis_family: str = DEFAULT_FAMILY
    # Per-record covariates the cockpit captures as free-text fields and exports as
    # their own columns (e.g. soil depth, fraction). Unlike ``moderators`` — which
    # are paper-level and broadcast to every record — these vary *row to row*, so a
    # nested table (one variable measured at several depths) keeps each depth on its
    # own record instead of jamming it into the group name. The pack declares the
    # field names; the human fills the value per record (locate never guesses them).
    record_covariates: list[str] = Field(default_factory=list)

    @field_validator("target_variables", mode="before")
    @classmethod
    def _coerce_target_variables(cls, value):
        if not isinstance(value, list):
            return value
        return [{"name": v} if isinstance(v, str) else v for v in value]

    @model_validator(mode="after")
    def _validate_pack(self) -> "TaskSpec":
        self.domain = self.domain.strip()
        if not self.domain:
            raise ValueError("task pack domain must not be blank")
        if not self.target_variables:
            raise ValueError("task pack must define at least one target variable")

        # Fail loudly on a typo'd family rather than silently collecting the wrong
        # shape; get_family raises KeyError listing the known names.
        family = get_family(self.analysis_family)

        seen: set[str] = set()
        for tv in self.target_variables:
            key = tv.name.casefold()
            if key in seen:
                raise ValueError(f"duplicate target variable name: {tv.name!r}")
            seen.add(key)

        # Record covariates become CSV columns, so a blank/duplicate name or one
        # that collides with a structural key, a family value field, or a moderator
        # would silently overwrite real data on export — reject it here instead.
        self.record_covariates = [c.strip() for c in self.record_covariates]
        taken = (
            {k.casefold() for k in RESERVED_RECORD_KEYS}
            | {r.casefold() for r in family.roles}
            | {m.casefold() for m in self.moderators}
        )
        cov_seen: set[str] = set()
        for cov in self.record_covariates:
            if not cov:
                raise ValueError("record_covariates entries must not be blank")
            key = cov.casefold()
            if key in cov_seen:
                raise ValueError(f"duplicate record covariate: {cov!r}")
            cov_seen.add(key)
            if key in taken:
                raise ValueError(
                    f"record covariate {cov!r} collides with a reserved column, a "
                    f"{self.analysis_family} value field, or a moderator name"
                )
        return self

    @property
    def variable_names(self) -> list[str]:
        """Canonical names only — for eval/benchmark code that compares to GT."""
        return [tv.name for tv in self.target_variables]

    def digest(self) -> str:
        """Stable short hash of the whole pack. Identifies the vocabulary that
        produced a cache / draft, so changing target metrics keys to a different
        file instead of mixing records across vocabularies."""
        data = self.model_dump(mode="json")
        # Keep the default family out of the digest so existing continuous packs
        # (written before analysis_family existed) keep their cache stamp and their
        # saved cockpit drafts. A non-default family does change the stamp, which is
        # what we want — binary and continuous records must not mix.
        if data.get("analysis_family") == DEFAULT_FAMILY:
            data.pop("analysis_family", None)
        # Same reasoning for record_covariates: an empty list is "no change", so it
        # stays out of the digest and existing packs keep their stamp; declaring
        # covariates does change the stamp (records gain columns — don't mix them).
        if not data.get("record_covariates"):
            data.pop("record_covariates", None)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    @property
    def cache_stamp(self) -> str:
        """``{domain}.{digest}`` infix shared by the extract cache and the review
        draft/log filenames, so both isolate by task pack the same way."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.domain.lower()).strip("_") or "taskpack"
        return f"{slug}.{self.digest()}"

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TaskSpec":
        """Load a task pack from a YAML file the researcher edits."""
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(
            self.model_dump(exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )


# Built-in fallback used when no task pack file is supplied. The real vocabulary
# lives in data/taskpacks/*.yaml; this just keeps locate() runnable out of the box.
SOIL_AGRI_SPEC = TaskSpec(
    domain="soil_agri_n2o",
    target_variables=[
        TargetVar(name="soil organic carbon", aliases=["SOC", "TOC", "organic carbon"], unit_hint="g/kg"),
        TargetVar(name="N2O emission", aliases=["N2O flux", "nitrous oxide emission"]),
        "grain yield", "N uptake", "CH4 flux", "CO2 flux", "soil mineral nitrogen",
    ],
    moderators=[
        "site/location", "mean annual temperature", "mean annual precipitation",
        "soil type", "soil texture", "initial pH", "experiment duration",
    ],
)


# --- what we ask Gemini to return (no geometry, just block_id choices) ---
class RawRegion(BaseModel):
    block_id: str
    variable_name: str
    candidate_structure: Optional[str] = None
    ambiguous: bool = False


class RawModerator(BaseModel):
    field: str
    value: str
    block_id: Optional[str] = None


class RawSampleSize(BaseModel):
    value: Optional[str] = None
    n_control: Optional[str] = None
    n_treatment: Optional[str] = None
    note: Optional[str] = None
    block_id: Optional[str] = None


class LocatorResponse(BaseModel):
    include: bool
    reason: Optional[str] = None
    target_variables_present: list[str] = Field(default_factory=list)
    moderators_present: list[str] = Field(default_factory=list)
    regions: list[RawRegion] = Field(default_factory=list)
    moderators: list[RawModerator] = Field(default_factory=list)
    sample_size: Optional[RawSampleSize] = None


class LocateOutput(BaseModel):
    screening: ScreeningResult
    regions: list[LocatedRegion]
    moderators: list[ExtractedModerator] = Field(default_factory=list)
    moderator_fields: list[str] = Field(default_factory=list)  # full set, for the cockpit panel
    sample_size: Optional[SampleSize] = None
    problems: list[str] = Field(default_factory=list)
    # Which analysis family this paper was located for; the cockpit builds its
    # input form from it. Defaults to continuous so caches written before families
    # existed render exactly as before.
    analysis_family: str = DEFAULT_FAMILY
    # Per-record covariate field names from the task pack (see TaskSpec); the
    # cockpit renders one input per name and exports each as its own column.
    # Empty on caches written before covariates existed -> the form is unchanged.
    record_covariates: list[str] = Field(default_factory=list)
    # Stamp of the task pack that produced this output; the cockpit keys review
    # progress by it so changing target metrics doesn't mix records. None on
    # caches written before stamping existed.
    task_stamp: Optional[str] = None


def build_payload(doc: SourceDoc) -> str:
    """Serialize blocks into a compact, addressable listing for the model."""
    lines = [
        "Addressable blocks (choose block_id values from this list only):",
        "",
    ]
    for b in doc.blocks:
        snippet = b.text[:_BLOCK_TEXT_CHARS].replace("\n", " ").strip()
        if b.kind == "figure":
            snippet = snippet or "(figure / image region)"
        lines.append(f"[{b.block_id}] p{b.page} {b.kind}: {snippet}")
    return "\n".join(lines)


def _system_instruction(spec: TaskSpec) -> str:
    family = get_family(spec.analysis_family)
    field_lines = "\n  - ".join(
        f"{field.role}: {field.label} ({field.arm} arm)"
        for field in family.fields
    )
    sample_size_instruction = (
        "4. SAMPLE SIZE: find the replicate count n behind the reported means. Look "
        "ONLY for a phrase that literally contains 'replicate', 'n = ', or "
        "'triplicate'/'duplicate'; the number attached to that word is n (e.g. "
        "'... x 3 replicates' -> n=3). Ignore counts of sites, sub-samples, plots, "
        "or total samples. Put the exact phrase in note, the integer in value, the "
        "block_id, and n_control/n_treatment if they differ. If no such phrase "
        "exists, leave sample_size null.\n"
    )
    if not family.has_sample_size:
        sample_size_instruction = (
            "4. SAMPLE SIZE: leave sample_size null. This family reads its totals "
            "from the located data table rather than from a separate replicate-count "
            "statement.\n"
        )

    return (
        "You are screening and locating data for a meta-analysis in the domain: "
        f"{spec.domain}.\n\n"
        "Analysis family for this task:\n"
        f"- {family.name}: {family.label}\n"
        "- A completed record in the cockpit will ask the human for these fields:\n  - "
        + field_lines
        + "\n\n"
        "Target response variables to look for (report each under its CANONICAL "
        "name, the bold name before any parentheses, even if the paper uses a "
        "synonym):\n  - "
        + "\n  - ".join(tv.as_prompt_line() for tv in spec.target_variables)
        + "\n\nModerator variables to note (paper level only):\n  - "
        + "\n  - ".join(spec.moderators)
        + "\n\nYour job has two parts and STRICT limits:\n"
        "1. SCREEN: decide include/exclude, and list which target variables and "
        "moderators are present in the paper.\n"
        "2. LOCATE: for each target variable you find, output the block_id of the "
        "block that contains its data (prefer tables over figures).\n"
        "3. MODERATORS: read each moderator field's paper/site-level value as "
        "printed (e.g. site name, MAP/MAT, soil type/texture, elevation, sampling "
        "depth, study years, climate), with the block_id where you read it. These "
        "are paper-level context, NOT per-row values — report each once.\n"
        + sample_size_instruction
        + "\n"
        "HARD RULES:\n"
        "- Only use block_id values that appear in the provided block list. Never "
        "invent a block_id or describe a location in prose.\n"
        "- Do NOT report any response-variable numbers. Do NOT pair treatment vs "
        "control. Only locate where the variable's data lives.\n"
        "- Prefer a block that contains enough information for the family fields "
        "listed above (for example, binary two-arm records need events and totals "
        "for both arms).\n"
        "- Moderator values ARE allowed (they are single stated facts); copy them "
        "verbatim and cite the block_id. Omit a moderator you cannot find.\n"
        "- If you are unsure a block really contains the variable, set "
        "ambiguous=true rather than omitting it.\n"
        "- In `variable_name`, return the canonical target name from the list "
        "above, NOT the paper's wording, so the same metric aligns across papers.\n"
        "- candidate_structure is an optional free-text hint about layout (e.g. "
        "'rows=treatments, cols=mean/sd/n'); it is a hint, not a commitment."
    )


def resolve_regions(
    doc: SourceDoc, raw: LocatorResponse, paper_id: str
) -> tuple[list[LocatedRegion], list[str]]:
    """Attach real geometry to model-chosen block_ids; drop+report invented ones.

    Pure and deterministic — this is the C1 guard, testable without the API.
    """
    regions: list[LocatedRegion] = []
    problems: list[str] = []
    seen: set[str] = set()

    for r in raw.regions:
        block = doc.block(r.block_id)
        if block is None:
            problems.append(
                f"unknown block_id {r.block_id!r} for variable "
                f"{r.variable_name!r} (dropped — model may have invented it)"
            )
            continue
        region_id = f"{paper_id}:{block.block_id}:{_slug(r.variable_name)}"
        if region_id in seen:
            continue
        seen.add(region_id)
        regions.append(
            LocatedRegion(
                region_id=region_id,
                paper_id=paper_id,
                variable_name=r.variable_name,
                citation=Citation(
                    block_id=block.block_id,
                    page=block.page,
                    bbox=block.bbox,
                    kind=block.kind,
                ),
                candidate_structure=r.candidate_structure,
                ambiguous=r.ambiguous,
            )
        )
    return regions, problems


def resolve_moderators(doc: SourceDoc, raw: LocatorResponse) -> list[ExtractedModerator]:
    """Keep each moderator value; attach a citation when its block_id is real."""
    mods: list[ExtractedModerator] = []
    seen: set[str] = set()
    for m in raw.moderators:
        if not m.value or _slug(m.field) in seen:
            continue
        seen.add(_slug(m.field))
        block = doc.block(m.block_id) if m.block_id else None
        citation = (
            Citation(block_id=block.block_id, page=block.page, bbox=block.bbox, kind=block.kind)
            if block is not None
            else None
        )
        mods.append(ExtractedModerator(field=m.field, value=m.value, citation=citation))
    return mods


def resolve_sample_size(doc: SourceDoc, raw: LocatorResponse) -> Optional[SampleSize]:
    """Keep the model's n statement; attach a citation when its block_id is real."""
    rs = raw.sample_size
    if rs is None or not (rs.value or rs.n_control or rs.n_treatment):
        return None
    block = doc.block(rs.block_id) if rs.block_id else None
    citation = (
        Citation(block_id=block.block_id, page=block.page, bbox=block.bbox, kind=block.kind)
        if block is not None
        else None
    )
    return SampleSize(
        value=rs.value, n_control=rs.n_control, n_treatment=rs.n_treatment,
        note=rs.note, citation=citation,
    )


def locate(
    doc: SourceDoc,
    spec: TaskSpec = SOIL_AGRI_SPEC,
    model: Optional[str] = None,
    client=None,
) -> LocateOutput:
    """Run Tier 1 + Tier 2 over an ingested SourceDoc via Gemini structured output."""
    from google.genai import types

    from .extractor import DEFAULT_MODEL, _build_client

    client = client or _build_client()
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=[build_payload(doc)],
        config=types.GenerateContentConfig(
            system_instruction=_system_instruction(spec),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_response_schema(),
        ),
    )
    parsed = getattr(response, "parsed", None)
    raw = (
        LocatorResponse.model_validate(parsed)
        if parsed is not None
        else LocatorResponse.model_validate_json(response.text)
    )

    regions, problems = resolve_regions(doc, raw, doc.doc_id)
    screening = ScreeningResult(
        paper_id=doc.doc_id,
        include=raw.include,
        reason=raw.reason,
        target_variables_present=raw.target_variables_present,
        moderators_present=raw.moderators_present,
    )
    return LocateOutput(
        screening=screening,
        regions=regions,
        moderators=resolve_moderators(doc, raw),
        moderator_fields=spec.moderators,
        sample_size=resolve_sample_size(doc, raw),
        problems=problems,
        task_stamp=spec.cache_stamp,
        analysis_family=spec.analysis_family,
        record_covariates=spec.record_covariates,
    )


def _response_schema() -> dict:
    """Inline $defs/$ref, which Gemini's response_schema does not accept."""
    schema = LocatorResponse.model_json_schema()
    defs = schema.pop("$defs", {})

    def _inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node.pop("$ref").split("/")[-1]
                node.update(_inline(defs[ref]))
            return {k: _inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_inline(v) for v in node]
        return node

    return _inline(schema)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
