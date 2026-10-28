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
from metaextract.highlight import Match, locate_value_in_span, locate_values  # noqa: E402
from metaextract.render import render_page, render_region  # noqa: E402
from metaextract.sampling import find_sample_size_candidates  # noqa: E402
from metaextract.sourcedoc import BBox, SourceBlock  # noqa: E402
from metaextract.records import Citation, LocatedRegion  # noqa: E402
from metaextract.tabular import make_pairings, parse_block  # noqa: E402
from metaextract.review_validation import (  # noqa: E402
    target_variable_coverage,
    validate_record,
    validate_records_for_export,
)

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
    try:
        family = get_family(getattr(out, "analysis_family", None))
    except KeyError as e:
        st.error(f"⚠ Invalid analysis family in cache: {e}. Re-run prepare_paper.py.")
        st.stop()
    # Per-record covariates declared by the task pack (e.g. depth, fraction). Empty
    # on caches prepared before covariates existed, so the form looks unchanged.
    covariates = list(getattr(out, "record_covariates", []) or [])
    # These fields form the row identity used to bind site/experiment/time
    # context.  Depth declared by a task pack joins the same identity rather
    # than being treated as a paper-level moderator.
    identity_fields = ["site", "experiment", "time", *covariates]
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
    coverage_status = draft.get("coverage_status", {})
    override_key = f"screening_override_{scope}"
    if override_key not in st.session_state:
        st.session_state[override_key] = bool(draft.get("screening_override", False))

    # --- Tier 1 screening summary ---
    s = out.screening
    st.sidebar.markdown(f"**Include:** {'✅' if s.include else '❌'}")
    st.sidebar.markdown(f"**Variables found:** {len(s.target_variables_present)}")
    st.sidebar.markdown(f"**Moderators:** {', '.join(s.moderators_present) or '—'}")
    st.sidebar.caption(f"📐 Family: {family.label}")
    expected_variables = list(getattr(out, "target_variables", []) or [])
    if not expected_variables:
        expected_variables = list(s.target_variables_present)
    coverage = target_variable_coverage(
        expected_variables,
        [r.variable_name for r in out.regions],
        screened_present=s.target_variables_present,
    )
    missing_coverage = [row for row in coverage if row["status"] != "located"]
    with st.sidebar.expander(
        f"Target coverage · {len(expected_variables) - len(missing_coverage)}/{len(expected_variables)} located",
        expanded=bool(missing_coverage),
    ):
        if coverage:
            st.dataframe(
                pd.DataFrame(coverage)[["variable_name", "status", "located_count"]],
                hide_index=True,
                width="stretch",
            )
        if missing_coverage:
            st.warning("Review every missing target before treating this paper as complete.")
    if out.problems:
        st.sidebar.error(f"{len(out.problems)} hallucinated block_id(s) dropped")
    st.sidebar.metric("Records assembled", len(records))

    if not out.screening.include:
        st.warning("🚫 The locator screened this paper out.")
        st.caption(out.screening.reason or "No screening reason was recorded.")
        st.session_state[override_key] = st.checkbox(
            "Override screening and review this paper manually",
            value=st.session_state[override_key],
            key=f"screening_override_checkbox_{scope}",
        )
        if not st.session_state[override_key]:
            st.info("The paper remains excluded until a reviewer overrides the decision.")
            reviewstore.save_draft(
                ROOT, study, records, saved_mods, stamp=stamp,
                screening_override=False, coverage_status=coverage_status,
            )
            return
        st.warning("Manual screening override is active; all missing target variables remain review items.")

    if not out.regions:
        # An excluded paper is the Tier-1 screen working, not a broken view — show
        # the verdict and the model's reason so it reads as "correctly skipped".
        st.info("No regions were located. Use the coverage table to identify missing targets, "
                "then re-run location or add a manual region from the source document.")
        missing_targets = [row["variable_name"] for row in coverage if row["status"] != "located"]
        paragraph_blocks = [b for b in doc.blocks if b.kind == "paragraph"]
        if missing_targets and paragraph_blocks:
            st.markdown("**Add a manually located target region**")
            target = st.selectbox("Target variable", missing_targets, key=f"manual_target_{scope}")
            source_block = st.selectbox(
                "Source block",
                paragraph_blocks,
                format_func=lambda b: f"{b.block_id} · p{b.page} · {b.text[:80]}",
                key=f"manual_block_{scope}",
            )
            if st.button("Add manual region", key=f"manual_region_{scope}"):
                slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in target).strip("_")
                out.regions.append(
                    LocatedRegion(
                        region_id=f"{doc.doc_id}:{source_block.block_id}:{slug}",
                        paper_id=doc.doc_id,
                        variable_name=target,
                        citation=Citation(
                            block_id=source_block.block_id,
                            page=source_block.page,
                            bbox=source_block.bbox,
                            kind=source_block.kind,
                        ),
                    )
                )
                reviewstore.append_log(
                    ROOT, study, "add_manual_region", variable_name=target,
                    records_after=len(records), stamp=stamp,
                )
                cockpit_cache.save(ROOT, study, doc, out)
                st.rerun()
        reviewstore.save_draft(
            ROOT, study, records, saved_mods, stamp=stamp,
            screening_override=st.session_state[override_key], coverage_status=coverage_status,
        )
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
    if block is None:
        st.error(
            f"⚠ Cached region refers to missing block {region.citation.block_id!r}. "
            "Re-run prepare_paper.py for this paper."
        )
        return

    # --- moderators (paper-level): AI suggests, human confirms + binds (v3.1 §2) ---
    mod_ai = {m.field: m.value for m in out.moderators}
    mod_fields = out.moderator_fields or list(mod_ai)
    bound_key = f"bound_mods_{scope}"
    if bound_key not in st.session_state:
        st.session_state[bound_key] = dict(saved_mods)
    with st.expander(f"📋 Moderators (paper-level suggestions) — {len(mod_ai)} AI-suggested"):
        st.caption(
            "AI suggestions are shown for reference. Edit values if needed, then click "
            "Confirm and bind; opening this page alone does not bind them."
        )
        mod_values: dict[str, str] = {}
        mcols = st.columns(3)
        for i, f in enumerate(mod_fields):
            suggestion = mod_ai.get(f)
            if suggestion:
                mcols[i % 3].caption(f"AI suggestion: {suggestion}")
            mod_values[f] = mcols[i % 3].text_input(
                f,
                value=st.session_state[bound_key].get(f, ""),
                key=f"mod_{scope}_{f}",
            )
        if st.button("Confirm and bind moderators", key=f"bind_mods_{scope}"):
            st.session_state[bound_key] = {
                k: v.strip() for k, v in mod_values.items() if v.strip()
            }
            reviewstore.append_log(
                ROOT, study, "bind_moderators", records_after=len(records), stamp=stamp
            )
            reviewstore.save_draft(
                ROOT,
                study,
                records,
                st.session_state[bound_key],
                stamp=stamp,
                screening_override=st.session_state.get(override_key, False),
                coverage_status=coverage_status,
            )
            st.rerun()
    bound = dict(st.session_state[bound_key])

    # --- sample size (n): deterministic finder surfaces candidates, human picks ---
    # (the LLM proved unreliable at n; regex over the source is not — v3.1: locate
    # reliably, human decides.)
    # Only families whose n comes from a replicate statement (continuous) use the
    # finder; families that read totals in-table (binary) skip this panel.
    default_n = ""
    n_source: dict[str, dict] = {}
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
            n_source = {
                "n_c": {
                    "source": "sample_size_finder",
                    "block_id": pick.block_id,
                    "page": pick.page,
                    "evidence_text": pick.evidence_text,
                },
                "n_t": {
                    "source": "sample_size_finder",
                    "block_id": pick.block_id,
                    "page": pick.page,
                    "evidence_text": pick.evidence_text,
                },
            }
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
        identity_values: dict[str, str] = {}
        st.markdown("**Record identity** — confirm the site, experiment, time, and depth for this row")
        icols = st.columns(min(len(identity_fields), 3))
        for i, identity_field in enumerate(identity_fields):
            identity_values[identity_field] = icols[i % len(icols)].text_input(
                identity_field,
                key=f"identity_{scope}_{region.region_id}_{identity_field}",
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
            funit, fdisp = st.columns(2)
            fig_unit = funit.text_input("Unit", key=f"funit_{scope}_{region.region_id}")
            fig_dispersion = (
                fdisp.selectbox("Dispersion", ["sd", "se"], key=f"fdisp_{scope}_{region.region_id}")
                if any(f.role.startswith("sd") for f in family.fields) else ""
            )
            pair = {
                "control_group": ctrl, "treatment_group": trt,
                "unit": fig_unit, "dispersion": fig_dispersion,
                **fig_entered, **covariate_values,
                "record_identity": identity_values,
                "identity_confirmed": False,
                "moderator_binding_status": "pending",
                "n_evidence": n_source,
                "source": "figure_digitized", "status": status, "digitizer_ref": digiref,
            }
            if n_source:
                pair["evidence"] = n_source
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
                        acol1, acol2 = st.columns(2)
                        auto_unit = acol1.text_input("Unit for pairings", key=f"auto_unit_{scope}")
                        auto_dispersion = acol2.selectbox(
                            "Dispersion for pairings", ["sd", "se"], key=f"auto_disp_{scope}"
                        )
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
                            width="stretch", height=140,
                        )
                        if st.button(
                            f"✓ Accept {len(cands)} pairings",
                            type="primary",
                            key=f"accept_pairings_{scope}_{region.region_id}",
                        ):
                            for r in cands:
                                rec = {k: v for k, v in r.items() if k != "_bboxes"}
                                rec["evidence"] = {
                                    role: {
                                        "source": "geometry_auto",
                                        "block_id": region.citation.block_id,
                                        "page": region.citation.page,
                                        "bbox": list(region.citation.bbox.as_tuple()),
                                        "evidence_text": rec[role],
                                    }
                                    for role in ("mean_c", "sd_c", "mean_t", "sd_t")
                                    if rec.get(role)
                                }
                                records.append({
                                    "paper_id": doc.doc_id, "variable_name": region.variable_name,
                                    "source_block_id": region.citation.block_id,
                                    "page": region.citation.page, "kind": region.citation.kind,
                                    "unit": auto_unit, "dispersion": auto_dispersion,
                                    "n_c": n_c_val, "n_t": n_t_val, **rec, **covariate_values,
                                    "evidence": {**rec.get("evidence", {}), **n_source},
                                    "n_evidence": n_source,
                                    "record_identity": dict(identity_values),
                                    "identity_confirmed": False,
                                    "moderator_binding_status": "pending",
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
                "record_identity": identity_values,
                "identity_confirmed": False,
                "moderator_binding_status": "pending",
            }
            if field_evidence:
                pair["evidence"] = field_evidence
            if n_source:
                pair["evidence"] = {**field_evidence, **n_source}
            if unmatched:
                st.warning(f"Not found on page {region.citation.page} (typo?): {', '.join(unmatched)}")

            if cands:  # candidate pairings take over the highlight
                values_to_highlight = [bb for r in cands for bb in r["_bboxes"]]

        if st.button("➕ Add record", type="primary", key=f"add_record_{scope}_{region.region_id}"):
            new_record = {
                "paper_id": doc.doc_id,
                "variable_name": region.variable_name,
                "source_block_id": region.citation.block_id,
                "page": region.citation.page,
                "kind": region.citation.kind,
                **pair,
            }
            record_flags = validate_record(new_record, family, source_doc=doc)
            new_record["qa_flags"] = "; ".join(record_flags)
            new_record["review_status"] = "needs_review" if record_flags else "ready"
            records.append(new_record)
            reviewstore.append_log(
                ROOT, study, "add_record",
                variable_name=region.variable_name, records_after=len(records),
                stamp=stamp,
            )
            if record_flags:
                st.warning(
                    "Record saved as a draft; export remains disabled until fixed: "
                    + ", ".join(record_flags)
                )
            st.rerun()

    # --- left: PDF with evidence highlighted ---
    with left:
        if region.citation.kind == "figure":
            png = _render_crop(str(_pdf_path(study)), region.citation.page, region.citation.bbox.as_tuple())
            st.image(png, width="stretch",
                     caption=f"page {region.citation.page} — figure region (digitize values from here)")
        else:
            outline = (region.citation.bbox.as_tuple(),)
            vals = tuple(b.as_tuple() for b in values_to_highlight)
            png = _render(str(_pdf_path(study)), region.citation.page, outline, vals)
            st.image(png, width="stretch",
                     caption=f"page {region.citation.page} — blue: located block · red: entered values")

    # --- assembled records + export ---
    if records:
        st.divider()
        st.subheader(f"Assembled records ({len(records)})")
        df = pd.DataFrame(records)
        st.caption(
            "Paper-level moderator suggestions are reference values. Each row must have its "
            "own confirmed identity and moderator binding before export."
        )
        st.dataframe(df, width="stretch", height=200)

        for i, record in enumerate(records):
            with st.expander(
                f"Row {i} identity and moderator binding · "
                f"{record.get('variable_name', '')} · {record.get('moderator_binding_status', 'pending')}",
                expanded=record.get("moderator_binding_status") != "bound",
            ):
                existing_identity = record.get("record_identity", {})
                identity_inputs: dict[str, str] = {}
                cols = st.columns(min(len(identity_fields), 3))
                for j, field in enumerate(identity_fields):
                    identity_inputs[field] = cols[j % len(cols)].text_input(
                        field,
                        value=str(existing_identity.get(field, "")),
                        key=f"row_identity_{scope}_{i}_{field}",
                    ).strip()
                moderator_inputs: dict[str, str] = {}
                existing_mods = record.get("moderators", {})
                mcols = st.columns(3)
                for j, field in enumerate(mod_fields):
                    moderator_inputs[field] = mcols[j % len(mcols)].text_input(
                        field,
                        value=str(existing_mods.get(field, bound.get(field, ""))),
                        key=f"row_mod_{scope}_{i}_{field}",
                    ).strip()
                if st.button(
                    "Confirm this row's identity and moderators",
                    key=f"confirm_row_binding_{scope}_{i}",
                ):
                    record["record_identity"] = identity_inputs
                    record["identity_confirmed"] = True
                    record["moderators"] = moderator_inputs
                    record["moderator_binding_status"] = "bound"
                    record["qa_flags"] = "; ".join(
                        validate_record(
                            record,
                            family,
                            source_doc=doc,
                            identity_fields=tuple(identity_fields),
                            moderator_fields=tuple(mod_fields),
                        )
                    )
                    reviewstore.append_log(
                        ROOT,
                        study,
                        "bind_record_moderators",
                        variable_name=record.get("variable_name", ""),
                        records_after=len(records),
                        detail=f"row={i}",
                        stamp=stamp,
                    )
                    st.rerun()

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

        export_rows, export_errors = validate_records_for_export(
            records,
            family,
            source_doc=doc,
            identity_fields=tuple(identity_fields),
            require_identity=True,
            moderator_fields=tuple(mod_fields),
        )
        if export_errors:
            st.error(
                "Export disabled: every record must pass validation. "
                + " | ".join(export_errors[:5])
            )
        else:
            export_df = pd.DataFrame(export_rows)
            st.download_button(
                "⬇ Export meta_ready.csv",
                export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{study}_meta_ready.csv",
                mime="text/csv",
            )

    # Autosave the draft on every run so work survives refresh / restart. Mutating
    # actions rerun before reaching here; the *following* run persists the new state.
    saved_at = reviewstore.save_draft(
        ROOT,
        study,
        records,
        bound,
        stamp=stamp,
        screening_override=st.session_state.get(override_key, False),
        coverage_status=coverage_status,
    )
    st.sidebar.caption(f"💾 Auto-saved · {len(records)} record(s) · {saved_at}")
    st.sidebar.caption(f"Draft: {reviewstore.draft_path(ROOT, study, stamp).name}")


if __name__ == "__main__":
    main()
