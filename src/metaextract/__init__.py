"""metaextract: LLM-assisted data extraction for ecological meta-analysis."""

from .extract import apply_grounding, extract_region, ground_verify
from .extractor import extract_from_pdf
from .families import FAMILIES, AnalysisFamily, RecordField, get_family
from .highlight import (
    Match,
    locate_value,
    locate_value_in_span,
    locate_values,
    numeric_in_text,
    text_contains,
)
from .ingest import IngestError, ingest_pdf
from .locate import SOIL_AGRI_SPEC, LocateOutput, TargetVar, TaskSpec, locate
from .model_profiles import ModelProfile, ModelProfileConfig, load_model_profiles
from .pipeline import run_folder
from .records import Citation, LocatedRegion, ScreeningResult
from .schema import ExtractionResult
from .sampling import find_sample_size_candidates
from .sourcedoc import BBox, SourceBlock, SourceDoc, Word
from .tabular import TableRow, make_pairings, parse_block
from .validator import validate_result

__all__ = [
    "extract_from_pdf",
    "run_folder",
    "ExtractionResult",
    "validate_result",
    # v3.1 cockpit foundation (M0a)
    "ingest_pdf",
    "IngestError",
    "SourceDoc",
    "SourceBlock",
    "BBox",
    "Word",
    "locate_value",
    "locate_values",
    "locate_value_in_span",
    "Match",
    "numeric_in_text",
    "text_contains",
    # v3.1 Tier 3: on-demand AI extraction behind grounding (M1)
    "extract_region",
    "ground_verify",
    "apply_grounding",
    # v3.1 screen + locate (M0b)
    "locate",
    "LocateOutput",
    "TaskSpec",
    "TargetVar",
    "SOIL_AGRI_SPEC",
    "ModelProfile",
    "ModelProfileConfig",
    "load_model_profiles",
    # analysis families (per-record field set)
    "AnalysisFamily",
    "RecordField",
    "get_family",
    "FAMILIES",
    "LocatedRegion",
    "ScreeningResult",
    "Citation",
    # multi-pairing assist (cockpit M0b+)
    "parse_block",
    "make_pairings",
    "TableRow",
    "find_sample_size_candidates",
]

__version__ = "0.1.0"
