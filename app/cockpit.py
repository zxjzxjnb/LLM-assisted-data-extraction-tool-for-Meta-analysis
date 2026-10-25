"""MetaExtract verification cockpit (Architecture v3.1 §3, M0b).

One screen: located regions on the right, the PDF with the evidence highlighted
on the left. Click a region → the page jumps and outlines the block; type a value
→ that exact number lights up in the source. AI located; the human verifies and
assembles records. Runs offline from the cache written by scripts/prepare_paper.py.

Launch:  streamlit run app/cockpit.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metaextract import cockpit_cache, reviewstore  # noqa: E402
from metaextract.agent import load_review_queue  # noqa: E402
from metaextract.families import get_family  # noqa: E402
from metaextract.highlight import Match, locate_value, locate_value_in_span, locate_values  # noqa: E402
from metaextract.render import render_page, render_region  # noqa: E402
from metaextract.sampling import find_sample_size_candidates  # noqa: E402
from metaextract.sourcedoc import BBox, SourceBlock  # noqa: E402
from metaextract.tabular import make_pairings, parse_block  # noqa: E402

st.set_page_config(page_title="MetaExtract cockpit", layout="wide")


@st.cache_data(show_spinner=False)
def _render(pdf_path: str, page: int, outline: tuple, values: tuple, dpi: int = 135) -> bytes:
    return render_page(
        pdf_path, page,
        outline=[BBox(x0=a, y0=b, x1=c, y1=d) for a, b, c, d in outline],
        values=[BBox(x0=a, y0=b, x1=c, y1=d) for a, b, c, d in values],
        dpi=dpi,
    )


@st.cache_data(show_spinner=False)
def _render_crop(pdf_path: str, page: int, bbox: tuple) -> bytes:
    return render_region(pdf_path, page, BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]))


def _figure_caption(doc, page):
    caps = [b.text for b in doc.blocks_on(page) if b.text.strip().lower().startswith("fig")]
    return caps[0] if caps else ""


def _evidence_payload(
    block: SourceBlock, match: Match, source: str, evidence_text: str | None = None
) -> dict:
    return {
        "source": source,
        "block_id": block.block_id,
        "page": block.page,
        "bbox": list(match.bbox.as_tuple()),
        "evidence_text": evidence_text or " ".join(w.text for w in match.words),
        "exact": match.exact,
    }


def _locate_on_page(doc, page, value):
    """Return every match on the page, including duplicates within one block."""
    out = []
    for b in doc.blocks_on(page):
        for m in locate_values(b, value):
            out.append({"block": b, "match": m})
    return out


def _match_label(hit: dict) -> str:
    ev = " ".join(w.text for w in hit["match"].words)
    return f"{hit['block'].block_id} · p{hit['block'].page} · {ev[:80]}"


def _value_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _pdf_path(study: str) -> Path:
    return ROOT / "data" / "sample_papers" / f"{study}.pdf"


def _agent_report_path(study: str) -> Path:
    """The conventional hand-off file written by ``metaextract agent --report``."""
    return ROOT / "data" / "outputs" / f"agent_run_{study}.json"


def main() -> None:
    st.title("MetaExtract — verification cockpit")

    entries = cockpit_cache.list_cache_entries(ROOT)
    if not entries:
        st.warning("No cached papers. Run `python scripts/prepare_paper.py P1` first.")
        return

    entry = st.sidebar.selectbox("Paper", entries, format_func=lambda e: e.label)
    study = entry.study
    try:
        doc, out = cockpit_cache.load(ROOT, study, stamp=entry.stamp)
    except cockpit_cache.CacheError as e:
        # A corrupt / outdated cache must not crash the page with a traceback — show
        # the one clear remedy (re-run prepare_paper) and stop this run cleanly.
        st.error(f"⚠ {e}")
        st.stop()
    # The analysis family drives the whole input form (which numbers a record
    # holds). Older caches have no family and default to continuous (today's form).
    family = get_family(getattr(out, "analysis_family", None))
    # Per-record covariates declared by the task pack (e.g. depth, fraction). Empty
    # on caches prepared before covariates existed, so the form looks unchanged.
    covariates = list(getattr(out, "record_covariates", []) or [])
    # Resume: seed this session's records from the saved draft (autosaved below).
    # Keyed by the task pack stamp so records assembled under different target
    # metrics never mix (matches the extract cache's filename stamping).
    stamp = out.task_stamp
    scope = f"{study}_{stamp or 'legacy'}"
    draft = reviewstore.load_draft(ROOT, study, stamp=stamp)
    # A corrupt draft is quarantined and reported only on the run that finds it
    # (it's renamed aside after), so latch the warning in session_state to keep it
    # visible for the rest of the session instead of flashing once.
    err_key = f"draft_error_{scope}"
    if draft.get("error"):
        st.session_state[err_key] = draft["error"]
    if st.session_state.get(err_key):
        st.error(f"⚠ {st.session_state[err_key]}")
    records = st.session_state.setdefault(f"records_{study}_{stamp}", draft["records"])
    saved_mods = draft.get("moderators", {})

    # --- Tier 1 screening summary ---
    s = out.screening
    st.sidebar.markdown(f"**Include:** {'✅' if s.include else '❌'}")
    st.sidebar.markdown(f"**Variables found:** {len(s.target_variables_present)}")
    st.sidebar.markdown(f"**Moderators:** {', '.join(s.moderators_present) or '—'}")
    st.sidebar.caption(f"📐 Family: {family.label}")
    if out.problems:
        st.sidebar.error(f"{len(out.problems)} hallucinated block_id(s) dropped")
    st.sidebar.metric("Records assembled", len(records))

    if not out.regions:
        # An excluded paper is the Tier-1 screen working, not a broken view — show
        # the verdict and the model's reason so it reads as "correctly skipped".
        if not out.screening.include:
            st.warning("🚫 This paper was screened **out** as off-topic for this meta-analysis.")
            if out.screening.reason:
                st.caption(f"Why: {out.screening.reason}")
            st.info("That is the screening step doing its job. Pick an included paper "
                    "(e.g. **P1**) from the sidebar to see located evidence.")
        else:
            st.info("This paper was included, but no regions were located. Try re-running "
                    "`python scripts/prepare_paper.py` for it, or pick another paper.")
        return

    labels = [f"{r.variable_name}  ·  {r.citation.block_id} (p{r.citation.page})" for r in out.regions]
    region_index = {r.region_id: i for i, r in enumerate(out.regions)}
    region_key = f"selected_region_{scope}"
    if region_key not in st.session_state:
        st.session_state[region_key] = 0

    # Agent flags are a hand-off, not an auto-acceptance path.  The reviewer gets
    # the reason and can jump directly to the cited region in this same cockpit.
    report_path = _agent_report_path(study)
    agent_queue = load_review_queue(report_path, study=study)
    if agent_queue.warning:
        st.sidebar.warning(f"🤖 {agent_queue.warning}")
    if agent_queue.flags:
        st.sidebar.warning(f"🤖 Agent review queue: {len(agent_queue.flags)} item(s)")
        with st.sidebar.expander("Open Agent flags", expanded=True):
            st.caption(f"Source: {report_path.name}")
            for i, flag in enumerate(agent_queue.flags):
                target = region_index.get(flag.region_id or "")
                title = flag.variable_name or flag.region_id or "Unspecified region"
                st.markdown(f"**{title}**  \n`{flag.reason}`")
                if flag.detail:
                    st.caption(flag.detail)
                if target is None:
                    st.caption("This flagged region is not in the current cached paper.")
                elif st.button("Open source region", key=f"agent_flag_{scope}_{i}"):
                    st.session_state[region_key] = target
                    st.rerun()

    idx = st.sidebar.radio(
        "Located regions", range(len(out.regions)),
        format_func=lambda i: labels[i], key=region_key,
    )
    region = out.regions[idx]
    block = doc.block(region.citation.block_id)

    # --- moderators (paper-level): AI suggests, human confirms + binds (v3.1 §2) ---
    mod_ai = {m.field: m.value for m in out.moderators}
    mod_fields = out.moderator_fields or list(mod_ai)
    with st.expander(f"📋 Moderators (paper-level) — {len(mod_ai)} AI-suggested · bind to all records"):
        st.caption(
            "AI suggested these paper/site-level values; edit/confirm. They bind to every "
            "exported record (v3.1 §2: AI may suggest, binding is human)."
        )
        mod_values: dict[str, str] = {}
        mcols = st.columns(3)
        for i, f in enumerate(mod_fields):
            default = saved_mods.get(f) or mod_ai.get(f, "")  # saved draft wins over AI suggestion
            mod_values[f] = mcols[i % 3].text_input(f, value=default, key=f"mod_{scope}_{f}")
    bound = {k: v.strip() for k, v in mod_values.items() if v.strip()}

    # --- sample size (n): deterministic finder surfaces candidates, human picks ---
    # (the LLM proved unreliable at n; regex over the source is not — v3.1: locate
    # reliably, human decides.)
    # Only families whose n comes from a replicate statement (continuous) use the
    # finder; families that read totals in-table (binary) skip this panel.
    default_n = ""
    if family.has_sample_size:
        n_cands = find_sample_size_candidates(doc)
        if n_cands:
            pick = n_cands[0]
            if len(n_cands) > 1:
                pi = st.selectbox(
                    "🧪 Sample size n — pick the replicate count",
                    range(len(n_cands)),
                    format_func=lambda k: f"n={n_cands[k].value} · {n_cands[k].phrase[:60]}",
                    key=f"npick_{scope}",
                )
                pick = n_cands[pi]
            default_n = pick.value
            st.caption(f"🧪 n = {pick.value} · source {pick.block_id} p{pick.page} — {pick.phrase}")
        else:
            st.caption("🧪 No sample-size statement found — enter n manually below.")

    left, right = st.columns([3, 2])

    # --- right: fill / assemble ---
    with right:
        st.subheader(region.variable_name)
        st.caption(f"{region.citation.kind} · {region.citation.block_id} · page {region.citation.page}")
        region_flags = [f for f in agent_queue.flags if f.region_id == region.region_id]
        if region_flags:
            with st.expander("🤖 Agent hand-off — review before adding a record", expanded=True):
                for flag in region_flags:
                    st.warning(f"{flag.reason}: {flag.detail or 'No further detail supplied.'}")
        if region.ambiguous:
            st.warning("Model marked this region ambiguous — verify it really holds this variable.")

        values_to_highlight: list[BBox] = []

        # Per-record covariates (e.g. depth, fraction): filled here, exported as
        # their own columns. They vary record to record, so — unlike the paper-level
        # moderators above — they live on the record, letting one variable measured
        # at several depths keep each depth on its own row instead of in the group
        # name. Rendered once here so manual, figure, and auto-pairing records share
        # the same values. The human fills them; locate never guesses them.
        covariate_values: dict[str, str] = {}
        if covariates:
            st.markdown("**Per-record covariates** — vary per record (e.g. depth, fraction)")
            ccols = st.columns(min(len(covariates), 3))
            for i, cov in enumerate(covariates):
                covariate_values[cov] = ccols[i % len(ccols)].text_input(
                    cov, key=f"cov_{scope}_{region.region_id}_{cov}"
                ).strip()

        if region.citation.kind == "figure":
            st.info("Figure — read values with a digitizer (WebPlotDigitizer); AI never reads chart values (v3.1 §6).")
            cap = _figure_caption(doc, region.citation.page)
            if cap:
                st.caption(f"📈 {cap[:240]}")
            status = st.selectbox(
                "Status", ["digitized", "located (not done)", "not in source"],
                key=f"fstatus_{scope}_{region.region_id}",
            )
            fc, ft = st.columns(2)
            ctrl = fc.text_input("Control group", key=f"fctrl_{scope}_{region.region_id}")
            trt = ft.text_input("Treatment group", key=f"ftrt_{scope}_{region.region_id}")
            g = st.columns(max(len(family.fields), 1))
            fig_entered: dict[str, str] = {}
            for j, fld in enumerate(family.fields):
                dflt = default_n if fld.is_n else ""
                fig_entered[fld.role] = g[j].text_input(
                    fld.label, value=dflt, key=f"f{fld.role}_{scope}_{region.region_id}"
                )
            digiref = st.text_input(
                "Digitizer ref (file / screenshot / axis calibration)", key=f"fref_{scope}_{region.region_id}"
            )
            pair = {
                "control_group": ctrl, "treatment_group": trt,
                **fig_entered, **covariate_values,
                "source": "figure_digitized", "status": status, "digitizer_ref": digiref,
            }
            entered = {}
        else:
            st.text_area("Block text (source)", block.text, height=80, disabled=True)

            # --- multi-pairing assist: geometry-based, control reused across treatments ---
            # Only continuous families: make_pairings assumes mean/SD columns.
            cand_key = f"cand_{scope}_{region.region_id}"
            cands = st.session_state.get(cand_key, [])
            if family.pairing:
                with st.expander("⚡ Auto-pairing — fast path for simple tables (variable-per-row)"):
                    st.caption(
                        "Only for plain tables where each row is one variable and columns are "
                        "treatments. Transposed/nested tables won't align — use manual entry below "
                        "(check the red highlights match)."
                    )
                    colspec = st.text_input(
                        "Columns left→right (comma-sep)", key=f"cols_{scope}",
                        placeholder="e.g. NF,AF,SL,FL",
                    )
                    col_labels = [c.strip() for c in colspec.split(",") if c.strip()]
                    n_c_val = n_t_val = default_n
                    if len(col_labels) >= 2:
                        control_col = st.selectbox("Control column", col_labels, key=f"ctrlcol_{scope}")
                        ncol, ecol = st.columns(2)
                        n_c_val = ncol.text_input("Nc (control n)", value=default_n, key=f"nc_{scope}")
                        n_t_val = ecol.text_input("Ne (treatment n)", value=default_n, key=f"ne_{scope}")
                        rows = parse_block(block, col_labels)
                        if not rows:
                            st.caption(
                                f"Couldn't align {len(col_labels)} columns here — fill manually below."
                            )
                        else:
                            if len(rows) == 1:
                                chosen = rows[0]
                            else:
                                ri = st.selectbox(
                                    "This block holds several rows — pick the one for this variable",
                                    range(len(rows)),
                                    format_func=lambda k: f"{rows[k].label}  →  "
                                    + ", ".join(
                                        f"{c}:{rows[k].cells[c].mean.text}"
                                        for c in col_labels if c in rows[k].cells
                                    ),
                                    key=f"row_{scope}_{region.region_id}",
                                )
                                chosen = rows[ri]
                            if st.button("Generate pairings", key=f"generate_pairings_{scope}_{region.region_id}"):
                                st.session_state[cand_key] = make_pairings(chosen.cells, col_labels, control_col)
                    cands = st.session_state.get(cand_key, [])
                    if cands:
                        st.dataframe(
                            [{k: v for k, v in r.items() if k != "_bboxes"} for r in cands],
                            use_container_width=True, height=140,
                        )
                        if st.button(
                            f"✓ Accept {len(cands)} pairings",
                            type="primary",
                            key=f"accept_pairings_{scope}_{region.region_id}",
                        ):
                            for r in cands:
                                rec = {k: v for k, v in r.items() if k != "_bboxes"}
                                records.append({
                                    "paper_id": doc.doc_id, "variable_name": region.variable_name,
                                    "source_block_id": region.citation.block_id,
                                    "page": region.citation.page, "kind": region.citation.kind,
                                    "unit": "", "dispersion": "",
                                    "n_c": n_c_val, "n_t": n_t_val, **rec, **covariate_values,
                                })
                            st.session_state[cand_key] = []
                            reviewstore.append_log(
                                ROOT, study, "accept_pairings",
                                variable_name=region.variable_name,
                                records_after=len(records), detail=f"{len(cands)} pairings",
                                stamp=stamp,
                            )
                            st.rerun()

            # --- Tier 3: on-demand AI extraction, behind the grounding gate (v3.1 §5.1) ---
            # The only path that produces AI numbers. Each value is ground-verified
            # against its own quoted span before it pre-fills the form; ungrounded
            # values are shown red and never auto-filled (the human must confirm).
            ai_key = f"ai_{scope}_{region.region_id}"
            with st.expander("🤖 AI extract (Tier 3) — on demand · grounded before shown"):
                st.caption(
                    "AI reads THIS block and must quote the source span for every value. "
                    "Each value is checked against that span and the block before it pre-fills "
                    "below. Red = could not be grounded — don't trust it, verify by hand."
                )
                if st.button("Extract this region with AI", key=f"aiget_{scope}_{region.region_id}"):
                    try:
                        from metaextract.extract import extract_region

                        slots = extract_region(block, region, family, region.variable_name)
                        st.session_state[ai_key] = [s.model_dump() for s in slots]
                        # pre-fill the manual-entry widgets only from grounded suggestions
                        grounded = 0
                        for s in slots:
                            if s.grounding_verified and s.value is not None:
                                st.session_state[f"{s.role}_{scope}_{region.region_id}"] = str(s.value)
                                grounded += 1
                        reviewstore.append_log(
                            ROOT, study, "ai_extract",
                            variable_name=region.variable_name, records_after=len(records),
                            detail=f"{grounded}/{len(slots)} grounded", stamp=stamp,
                        )
                        st.rerun()
                    except Exception as e:  # missing API key / network — stay usable offline
                        st.error(f"AI extract unavailable: {e}")
                for s in st.session_state.get(ai_key, []):
                    ev = (s.get("citation") or {}).get("evidence_text") or ""
                    if s["grounding_verified"]:
                        st.success(f"✓ {s['role']} = {s['value']} · grounded in “{ev}”")
                    else:
                        st.error(
                            f"⚠ {s['role']} = {s['value']} · UNVERIFIED "
                            f"({', '.join(s['flags'])}) — verify manually"
                        )

            st.markdown("**Manual entry** — works for any table layout (transposed, nested, …)")
            c1, c2 = st.columns(2)
            ctrl = c1.text_input("Control group", key=f"ctrl_{scope}_{region.region_id}")
            trt = c2.text_input("Treatment group", key=f"trt_{scope}_{region.region_id}")
            entered = {}
            field_evidence: dict[str, dict] = {}
            unmatched = []
            # grounded AI suggestions, by role — used to highlight a value at its
            # OWN cited block first (its real evidence) rather than scanning the
            # whole page, which would also light up any duplicate of that number.
            ai_grounded = {
                s["role"]: s
                for s in st.session_state.get(ai_key, [])
                if s["grounding_verified"] and s["value"] is not None
            }
            cols = st.columns(3)
            for j, fld in enumerate(family.fields):
                dflt = default_n if fld.is_n else ""
                v = cols[j % 3].text_input(
                    fld.label, value=dflt, key=f"{fld.role}_{scope}_{region.region_id}"
                )
                entered[fld.role] = v
                # highlight table values in the source; n-style values come from the
                # methods, not the table, so they are not searched for.
                if v.strip() and fld.highlight:
                    s = ai_grounded.get(fld.role)
                    citation = s.get("citation") if s else {}
                    cited = doc.block((citation or {}).get("block_id", "")) if s else None
                    evidence_text = (citation or {}).get("evidence_text") or ""
                    span_hit = (
                        locate_value_in_span(cited, v.strip(), evidence_text)
                        if cited is not None and str(s["value"]) == v.strip() and evidence_text
                        else None
                    )
                    if span_hit is not None:
                        values_to_highlight.append(span_hit.bbox)  # precise: AI's own evidence
                        field_evidence[fld.role] = _evidence_payload(
                            cited, span_hit, "ai_grounded", evidence_text=evidence_text
                        )
                    else:
                        hits = _locate_on_page(doc, region.citation.page, v.strip())
                        if len(hits) == 1:
                            chosen = hits[0]
                            values_to_highlight.append(chosen["match"].bbox)
                            field_evidence[fld.role] = _evidence_payload(
                                chosen["block"], chosen["match"], "manual"
                            )
                        elif len(hits) > 1:
                            preferred = next(
                                (
                                    k for k, hit in enumerate(hits)
                                    if hit["block"].block_id == region.citation.block_id
                                ),
                                0,
                            )
                            choice = st.selectbox(
                                f"Evidence for {fld.label} ({len(hits)} matches)",
                                range(len(hits)),
                                index=preferred,
                                format_func=lambda k: _match_label(hits[k]),
                                key=(
                                    f"ev_{fld.role}_{scope}_{region.region_id}_"
                                    f"{_value_key(v.strip())}"
                                ),
                            )
                            chosen = hits[choice]
                            values_to_highlight.append(chosen["match"].bbox)
                            field_evidence[fld.role] = _evidence_payload(
                                chosen["block"], chosen["match"], "manual_selected"
                            )
                        else:
                            unmatched.append(fld.label)
            u1, u2 = st.columns(2)
            unit = u1.text_input("Unit", key=f"unit_{scope}_{region.region_id}")
            # Dispersion (SD vs SE) only means something for families with an SD field.
            has_sd = any(f.role.startswith("sd") for f in family.fields)
            dispersion = (
                u2.selectbox("Dispersion", ["sd", "se"], key=f"disp_{scope}_{region.region_id}")
                if has_sd else ""
            )
            pair = {
                "control_group": ctrl,
                "treatment_group": trt,
                "unit": unit,
                "dispersion": dispersion,
                **entered,
                **covariate_values,
            }
            if field_evidence:
                pair["evidence"] = field_evidence
            if unmatched:
                st.warning(f"Not found on page {region.citation.page} (typo?): {', '.join(unmatched)}")

            if cands:  # candidate pairings take over the highlight
                values_to_highlight = [bb for r in cands for bb in r["_bboxes"]]

        if st.button("➕ Add record", type="primary", key=f"add_record_{scope}_{region.region_id}"):
            records.append({
                "paper_id": doc.doc_id,
                "variable_name": region.variable_name,
                "source_block_id": region.citation.block_id,
                "page": region.citation.page,
                "kind": region.citation.kind,
                **pair,
            })
            reviewstore.append_log(
                ROOT, study, "add_record",
                variable_name=region.variable_name, records_after=len(records),
                stamp=stamp,
            )
            st.rerun()

    # --- left: PDF with evidence highlighted ---
    with left:
        if region.citation.kind == "figure":
            png = _render_crop(str(_pdf_path(study)), region.citation.page, region.citation.bbox.as_tuple())
            st.image(png, use_container_width=True,
                     caption=f"page {region.citation.page} — figure region (digitize values from here)")
        else:
            outline = (region.citation.bbox.as_tuple(),)
            vals = tuple(b.as_tuple() for b in values_to_highlight)
            png = _render(str(_pdf_path(study)), region.citation.page, outline, vals)
            st.image(png, use_container_width=True,
                     caption=f"page {region.citation.page} — blue: located block · red: entered values")

    # --- assembled records + export ---
    if records:
        st.divider()
        st.subheader(f"Assembled records ({len(records)})")
        status = "bound" if bound else "unbound"
        df = pd.DataFrame(
            [{**bound, **r, "moderator_binding_status": status} for r in records]
        )
        if not bound:
            st.caption("⚠ No moderators bound yet — fill the Moderators panel to bind them.")
        st.dataframe(df, use_container_width=True, height=200)

        # delete a record — needed to fix mistakes when resuming across sessions
        dcol, bcol = st.columns([4, 1])
        del_i = dcol.selectbox(
            "Remove a record", range(len(records)),
            format_func=lambda i: f"{i}: {records[i].get('variable_name', '')} · "
            f"{records[i].get('treatment_group', '')} vs {records[i].get('control_group', '')}",
            key=f"del_{scope}",
        )
        if bcol.button("🗑 Remove", key=f"remove_{scope}"):
            removed = records.pop(del_i)
            reviewstore.append_log(
                ROOT, study, "delete_record",
                variable_name=removed.get("variable_name", ""), records_after=len(records),
                stamp=stamp,
            )
            st.rerun()

        st.download_button(
            "⬇ Export meta_ready.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{study}_meta_ready.csv",
            mime="text/csv",
        )

    # Autosave the draft on every run so work survives refresh / restart. Mutating
    # actions rerun before reaching here; the *following* run persists the new state.
    if records or bound:
        saved_at = reviewstore.save_draft(ROOT, study, records, bound, stamp=stamp)
        st.sidebar.caption(f"💾 Auto-saved · {len(records)} record(s) · {saved_at}")
    st.sidebar.caption(f"Draft: {reviewstore.draft_path(ROOT, study, stamp).name}")


if __name__ == "__main__":
    main()
