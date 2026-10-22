"""Gemini-based extraction of meta-analysis data from a PDF.

Design notes (and what changed vs. the original notebook):

*   **The whole PDF is sent to the model**, not PyPDF2-flattened text. Gemini's
    native document understanding reads tables and figure captions in place,
    which is where almost all meta-analysis numbers live. The old approach ran
    ``re.sub(r"\\s+", " ", text)`` over the page text, which collapsed exactly
    the row/column structure the data depends on.
*   **Structured output** (``response_schema``) forces valid JSON, so the
    fragile ``lstrip("```json")`` cleanup is gone.
*   The model is configurable and current; the notebook hard-coded
    ``gemini-1.0-pro-vision``, which has since been retired.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import ExtractionResult, gemini_response_schema

if TYPE_CHECKING:
    from .locate import TaskSpec

DEFAULT_MODEL = os.environ.get("METAEXTRACT_MODEL", "gemini-2.5-pro")

SYSTEM_INSTRUCTION = """\
You are a meticulous soil-ecology researcher performing data extraction for a \
quantitative meta-analysis. You read a primary research paper and extract every \
paired treatment-vs-control measurement, together with the study-level \
moderators needed to explain heterogeneity.

Rules:
1. Report numbers exactly as printed. Never invent, round, or unit-convert.
2. Treatment data must be paired with a control. If a treatment has no
   corresponding control measurement, omit that data point entirely.
3. For every response-variable data point, record where you read it (the
   `context.source`, e.g. "Table 2" or "Figure 3").
4. If a value is genuinely not reported in the paper, use null. Do not guess.
5. Prefer values from data tables over values read off figures; only use
   figures when the paper reports the number nowhere else.
"""

USER_PROMPT = (
    "Extract the meta-analysis data from the attached paper, following the "
    "required schema exactly."
)


# schema.py models a fixed, soil-specific moderator set. Task pack moderators are
# surfaced only for domains that schema actually covers; for any other domain they
# have no schema slot and would be silently dropped, so we omit them rather than
# mislead the model. Extend this set when schema.py gains broader moderator coverage.
SCHEMA_NATIVE_DOMAINS = {"soil_agri_n2o", "soil_C_fractions_landuse"}


def taskpack_instruction(task_spec: "TaskSpec | None") -> str:
    """Render a task pack as extraction guidance without changing the fixed schema.

    Target-variable guidance applies to any domain (``variable_name`` is free
    text, so the canonical name is honored regardless). Moderator guidance is
    emitted only for domains the fixed schema covers (:data:`SCHEMA_NATIVE_DOMAINS`);
    for other domains the pack's moderators have no schema slot, so listing them
    would mislead the model.
    """
    if task_spec is None:
        return ""

    lines = [
        "",
        "Task pack for this run:",
        f"- Domain: {task_spec.domain}",
        "- Only extract response variables that match these canonical target names",
        "  or their aliases. Return the canonical name in `variable_name`:",
    ]
    lines.extend(f"  - {tv.as_prompt_line()}" for tv in task_spec.target_variables)

    if task_spec.moderators and task_spec.domain in SCHEMA_NATIVE_DOMAINS:
        lines.extend([
            "- Prioritize these paper-level moderators when they fit the schema:",
            "  - " + "\n  - ".join(task_spec.moderators),
        ])
    return "\n".join(lines)


def _build_client(vertex: bool | None = None):
    """Create a google-genai client for either Vertex AI or the Gemini API.

    Vertex is used when ``GOOGLE_GENAI_USE_VERTEXAI`` / project env vars are set
    (matching the original Workbench setup); otherwise an API key is used.
    """
    from google import genai

    if vertex is None:
        vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {
            "1",
            "true",
        } or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))

    if vertex:
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def extract_from_pdf(
    pdf_path: str | Path,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    client=None,
    task_spec: "TaskSpec | None" = None,
) -> ExtractionResult:
    """Run extraction on a single PDF and return a validated result.

    Raises on API/parse failure so the caller (the batch pipeline) can record
    the paper as failed rather than silently dropping it.
    """
    from google.genai import types

    pdf_path = Path(pdf_path)
    client = client or _build_client()

    pdf_part = types.Part.from_bytes(
        data=pdf_path.read_bytes(),
        mime_type="application/pdf",
    )

    response = client.models.generate_content(
        model=model,
        contents=[pdf_part, USER_PROMPT],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION + taskpack_instruction(task_spec),
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=gemini_response_schema(),
        ),
    )

    # response.parsed is populated when the schema is satisfied; fall back to
    # validating the raw text so a near-miss is still recoverable.
    if getattr(response, "parsed", None) is not None:
        return ExtractionResult.model_validate(response.parsed)
    return ExtractionResult.model_validate_json(response.text)
