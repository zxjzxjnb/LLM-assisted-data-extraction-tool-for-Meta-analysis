"""Addressable source document — the bbox-carrying foundation of the cockpit.

Architecture v3.1 §3/§4 (C1): the cockpit's make-or-break feature is
*click a slot → jump to the exact spot in the PDF and highlight it*. That only
works if geometry comes from **deterministic ingest**, never from the LLM. The
Tier-2 locator later picks a ``block_id`` from this document; it does not invent
a location.

Two levels of geometry, both from PyMuPDF:

* **blocks** — paragraph / figure regions (and, as it happens, each *row* of a
  borderless journal table arrives as its own block). The jump target.
* **words** — every token with its own bbox. The *precise highlight* target: to
  confirm ``mean_t = 0.392`` we light up that one number, not a whole cell.

Word-level addressing replaces the cell-grid idea from v3.0: ``find_tables()``
reliably returns nothing on borderless journal tables (no ruling lines), so a
deterministic cell grid is not obtainable — but every number still has an exact
word bbox, which is what verification actually needs.

Coordinates are PDF points (1/72 inch), origin top-left, y increasing downward.
``SourceDoc.page_sizes`` lets a viewer scale bboxes to any render resolution.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BlockKind = Literal["paragraph", "figure"]


class BBox(BaseModel):
    """A rectangle in PDF points (top-left origin)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_rect(cls, rect) -> "BBox":
        """Build from a PyMuPDF ``Rect`` or a 4-tuple ``(x0, y0, x1, y1)``."""
        x0, y0, x1, y1 = (
            (rect.x0, rect.y0, rect.x1, rect.y1) if hasattr(rect, "x0") else rect
        )
        return cls(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def contains_center_of(self, other: "BBox") -> bool:
        cx, cy = (other.x0 + other.x1) / 2, (other.y0 + other.y1) / 2
        return self.x0 <= cx <= self.x1 and self.y0 <= cy <= self.y1

    @staticmethod
    def union(boxes: list["BBox"]) -> Optional["BBox"]:
        if not boxes:
            return None
        return BBox(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )


class Word(BaseModel):
    """A single token with its own geometry — the precise-highlight unit."""

    text: str
    bbox: BBox


class SourceBlock(BaseModel):
    """An addressable region of the source document (the jump target)."""

    block_id: str  # stable: "p1.b3", "p4.fig1"
    kind: BlockKind
    page: int  # 1-indexed
    bbox: BBox
    text: str = ""  # whitespace-normalized block text
    words: list[Word] = Field(default_factory=list)


class SourceDoc(BaseModel):
    """A whole PDF parsed into addressable blocks with geometry."""

    doc_id: str
    n_pages: int
    page_sizes: dict[int, tuple[float, float]] = Field(default_factory=dict)  # page -> (w, h)
    blocks: list[SourceBlock] = Field(default_factory=list)
    source_sha256: Optional[str] = None

    def block(self, block_id: str) -> Optional[SourceBlock]:
        return next((b for b in self.blocks if b.block_id == block_id), None)

    def blocks_on(self, page: int) -> list[SourceBlock]:
        return [b for b in self.blocks if b.page == page]
