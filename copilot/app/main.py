"""Streamlit UI entrypoint for the primer design copilot.

Run with: ``streamlit run copilot/app/main.py`` (by project convention).

``streamlit`` is an optional dependency (see the ``copilot`` extra in
``pyproject.toml``), so the import is deferred into ``render_app()`` rather
than performed at module scope — this module must remain importable without
``streamlit`` installed (e.g. for a plain import smoke-test).

Two tabs:
  * **Design (agent)** — a provider-agnostic LLM (OpenAI-style default, or
    Anthropic, or any OpenAI-compatible endpoint incl. local Ollama) designs
    primers by orchestrating the real tools.
  * **Score manual primers** — paste your own primers and score them directly
    with the trained head-A models (P(amplify) + efficiency); no LLM involved.
"""

from __future__ import annotations

# Known model ids per provider for the Design-tab dropdown. "(custom…)" reveals
# a free-text box for any other id (e.g. an Ollama/vLLM model via a base URL).
PROVIDER_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-fable-5",
    ],
}
_CUSTOM = "(custom…)"

# An in-domain example (IGHV1-2*02 template + two primers: a perfect match and a
# 3'-mismatched one) so manual scoring works out of the box without the agent.
_DEMO_TEMPLATE = (
    "ATGGACTGGACCTGGAGGATCCTCTTCTTGGTGGCAGCAGCCACAGGAGCCCACTCCCAGGTGCAGCTGGTGCAGT"
    "CTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTCACCGGCTA"
    "CTATATGCACTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGATGGATCAACCCTAACAGTGGTGGC"
    "ACAAACTATGCACAGAAGTTTCAGGGCAGGGTCACCATGACCAGGGACACGTCCATCAGCACAGCCTACATGGAGC"
    "TGAGCAGGCTGAGATCTGACGACACGGCCGTGTATTACTGTGCGAGAGA"
)
_DEMO_MANUAL_PRIMERS = "ATCCTCTTCTTGGTGGCAGC\nATCCTCTTCTTGGTGGCAAAA"  # perfect + 3'-mismatch

# Specificity demo (local IGHV DB) and the caveat that a count is only as
# trustworthy as the DB is representative.
_DEMO_PRIMER = "ATCCTCTTCTTGGTGGCAGC"  # a leader/FR1 primer for IGHV1-2*02
_DEMO_DB = "data/blast/ighv_db"
_DEMO_EXCLUDE = "IGHV1-2_02"
_EXCLUDE_HINT = (
    "The database subject id of the intended target, excluded from the off-target "
    "count. For the IGHV demo DB, use the sanitized allele id, e.g. IGHV1-2_02."
)
_SPEC_CAVEAT = (
    "⚠️ **An off-target count is only as trustworthy as the database is "
    "representative.** BLAST can only find hits that exist in the DB. This demo DB "
    "holds *only* IGHV alleles, so a primer for anything else returns **0 — which "
    "is uninformative, not evidence of specificity.** For a genome-scale answer use "
    "a transcriptome/genome DB or the remote NCBI option. See "
    "`docs/specificity_notes.md`."
)


def _clean_dna(text: str) -> str:
    """Normalize a pasted sequence: drop FASTA headers, whitespace, and case.

    Lets users paste a raw sequence, a wrapped multi-line block, or even a FASTA
    record (``>header`` lines are dropped) without breaking the tools.
    """
    body = "".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(">"))
    return "".join(body.split()).upper()


def render_app() -> None:
    """Render the Streamlit app (design-agent tab + manual-scoring tab).

    Requires ``streamlit``, imported lazily inside this function so the module
    remains importable without the optional ``copilot`` extra installed.

    Raises:
        ImportError: If ``streamlit`` is not installed.
    """
    import streamlit as st

    from copilot.agent.loop import run_agent_loop
    from copilot.agent.providers import get_provider
    from primer_core.tools.score import score_dual_head
    from primer_core.tools.specificity import check_specificity

    st.set_page_config(page_title="Primer Design Copilot", page_icon="🧬")
    st.title("🧬 Primer Design Copilot")
    st.caption(
        "Every sequence and number comes from a real tool (Primer3, ViennaRNA, "
        "BLAST, and the trained head-A models) — never invented."
    )

    with st.sidebar:
        st.header("Model artifacts (head A)")
        classifier_path = st.text_input(
            "Classifier (P(amplify))", value="data/models/head_a_classifier.joblib"
        )
        regressor_path = st.text_input(
            "Regressor (efficiency)", value="data/models/head_a_regressor.joblib"
        )
        st.header("Template")
        template = st.text_area(
            "Template sequence (5'->3', ACGT)",
            height=120,
            help="Required for primer-template annealing features (head A).",
            key="template",
        )
        st.header("Backend (Design tab)")
        provider_name = st.selectbox("Provider", list(PROVIDER_MODELS), index=0)
        model_choice = st.selectbox("Model", PROVIDER_MODELS[provider_name] + [_CUSTOM], index=0)
        model = (
            st.text_input("Custom model id", value="", placeholder="e.g. llama3.1 (Ollama)")
            if model_choice == _CUSTOM
            else model_choice
        )
        # Only the compatible-endpoint provider accepts a custom URL; show the
        # field only then, so its label needn't name any provider.
        base_url = ""
        if provider_name == "openai":
            base_url = st.text_input(
                "Custom API endpoint (blank = default)",
                value="",
                help=(
                    "For a local or self-hosted model with a compatible API — e.g. an "
                    "Ollama/vLLM server at http://localhost:11434/v1. Leave blank for the "
                    "standard hosted endpoint."
                ),
            )

    ctx = {
        "template_seq": _clean_dna(template) or None,
        "classifier_path": classifier_path.strip() or None,
        "regressor_path": regressor_path.strip() or None,
    }

    def _load_example() -> None:
        # Callbacks run before widgets re-instantiate, so setting the keyed
        # session_state fills the sidebar template + the manual primers box.
        st.session_state["template"] = _DEMO_TEMPLATE
        st.session_state["primers"] = _DEMO_MANUAL_PRIMERS

    tab_design, tab_manual, tab_spec = st.tabs(
        ["Design (agent)", "Score manual primers", "Specificity (BLAST)"]
    )

    # --- Tab 1: agent designs + scores ---------------------------------------
    with tab_design:
        if not ctx["template_seq"]:
            st.info(
                "Paste a **template sequence** in the sidebar — the agent designs primers for it."
            )
        goal = st.text_area(
            "Design goal",
            placeholder="e.g. qPCR primers for the loaded template, amplicon 75-150 bp, Tm ~60C",
            height=120,
        )
        design_blast = {
            "blast_db": None,
            "blast_remote": False,
            "blast_database": "nt",
            "blast_exclude": None,
        }
        with st.expander("Off-target BLAST (optional) — choose a database"):
            spec_mode = st.radio(
                "Off-target check",
                ["Off", "Local BLAST DB", "Remote NCBI"],
                horizontal=True,
                key="design_spec_mode",
            )
            if spec_mode == "Local BLAST DB":
                design_blast["blast_db"] = (
                    st.text_input(
                        "makeblastdb basename",
                        value="",
                        placeholder=_DEMO_DB,
                        key="design_blast_db",
                    ).strip()
                    or None
                )
            elif spec_mode == "Remote NCBI":
                design_blast["blast_remote"] = True
                design_blast["blast_database"] = (
                    st.text_input("NCBI database", value="nt", key="design_blast_ncbi").strip()
                    or "nt"
                )
                st.caption("⏳ Remote BLAST is slow/rate-limited — adds ~30 s-minutes per primer.")
            if spec_mode != "Off":
                design_blast["blast_exclude"] = (
                    st.text_input(
                        "Exclude on-target id (optional)",
                        key="design_blast_exclude",
                        help=_EXCLUDE_HINT,
                    ).strip()
                    or None
                )
                st.caption(
                    "A count is only meaningful if the DB represents the real off-target "
                    "space — see the Specificity tab."
                )
        design_clicked = st.button("Design", type="primary", disabled=not ctx["template_seq"])
        if design_clicked and not goal.strip():
            st.warning("Enter a design goal.")
        elif design_clicked and not (model or "").strip():
            st.warning("Pick a model (or enter a custom id).")
        elif design_clicked:
            kwargs = {"model": model}
            if provider_name == "openai" and base_url.strip():
                kwargs["base_url"] = base_url.strip()
            goal_msg = (
                f"{goal.strip()}\n\n"
                "The design template is already loaded by the runtime — call "
                "design_primers to get candidates (you do not need to supply the "
                "template sequence). Then call score_candidate on each candidate, flag "
                "risks, rank them best-first, and return the ranked-output JSON plus a "
                "short memo. Use sensible defaults; do not ask for clarification."
            )
            try:
                provider = get_provider(provider_name, **kwargs)
                with st.spinner("Designing and scoring candidates..."):
                    result = run_agent_loop(
                        goal_msg, provider=provider, context={**ctx, **design_blast}
                    )
            except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                st.error(f"Agent run failed: {exc}")
            else:
                st.subheader("Ranked candidates")
                st.dataframe(
                    [c.model_dump() for c in result["ranked_output"].candidates],
                    use_container_width=True,
                )
                st.subheader("Design memo")
                st.markdown(result["design_memo"] or "_(no memo returned)_")

    # --- Tab 2: score manually-entered primers (no LLM) ----------------------
    with tab_manual:
        st.markdown(
            "Score primers against the **template in the sidebar** (required for the "
            "annealing features) — no agent/LLM involved. One primer per line."
        )
        st.button(
            "Load IGHV example (template + primers)",
            on_click=_load_example,
            key="load_manual_example",
        )
        if not ctx["template_seq"]:
            st.info(
                "No template set — paste one in the sidebar, or click the example button above."
            )
        primers_text = st.text_area(
            "Primers (one per line)",
            placeholder="ATCCTCTTCTTGGTGGCAGC\nCTGAGGTGAAGAAGCCTGGG",
            height=120,
            key="primers",
        )
        if st.button("Score primers", key="score_btn"):
            primers = [s for s in (_clean_dna(p) for p in primers_text.splitlines()) if s]
            if not ctx["template_seq"]:
                st.warning(
                    "Enter a template sequence in the sidebar (needed for annealing features)."
                )
            elif not primers:
                st.warning("Enter at least one primer.")
            else:
                rows, caveat = [], None
                for p in primers:
                    try:
                        out = score_dual_head(
                            p,
                            template_seq=ctx["template_seq"],
                            classifier_path=ctx["classifier_path"],
                            regressor_path=ctx["regressor_path"],
                        )
                    except Exception as exc:  # noqa: BLE001 - report per primer
                        rows.append({"primer": p, "error": str(exc)})
                        continue
                    f = out.get("features", {})
                    caveat = out.get("caveats", [caveat])[0]
                    ci = out.get("efficiency_interval")
                    rows.append(
                        {
                            "primer": p,
                            "mismatch_count": f.get("mismatch_count"),
                            "annealing_dg": round(f.get("annealing_dg", float("nan")), 2),
                            "P(amplify)": out.get("amplify_probability"),
                            "predicted_efficiency": out.get("predicted_efficiency"),
                            "eff 90% interval": f"[{ci[0]}, {ci[1]}]" if ci else None,
                            "notes": "; ".join(out.get("notes", [])),
                        }
                    )
                st.dataframe(rows, use_container_width=True)
                if caveat:
                    st.caption(f"⚠️ {caveat}")
                st.caption(
                    "P(amplify) is isotonic-calibrated; 'eff 90% interval' is a conformal "
                    "prediction interval — read the interval, not the point estimate."
                )

    # --- Tab 3: BLAST specificity demo (local IGHV DB) -----------------------
    with tab_spec:
        st.warning(_SPEC_CAVEAT)
        st.markdown(
            f"**Demo:** a conserved leader/FR1 primer for **IGHV1-2\\*02** "
            f"(`{_DEMO_PRIMER}`) checked against the local IGHV allele DB "
            f"(`{_DEMO_DB}`), excluding the on-target allele. Build the DB first with "
            "`python scripts/build_ighv_blastdb.py`."
        )
        if st.button("Run IGHV specificity demo", key="spec_demo"):
            try:
                rep = check_specificity(
                    _DEMO_PRIMER, db_path=_DEMO_DB, exclude_target_id=_DEMO_EXCLUDE
                )
            except Exception as exc:  # noqa: BLE001 - surface build/install hints
                st.error(
                    f"Could not run demo: {exc}\n\nEnsure BLAST+ is installed and the DB "
                    "is built (`python scripts/build_ighv_blastdb.py`)."
                )
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("off-target hits", rep["off_target_hit_count"])
                c2.metric("best off-target bitscore", f"{rep['best_off_target_bitscore']:.1f}")
                c3.metric(
                    "3'-offtarget complementarity",
                    f"{rep['three_prime_offtarget_complementarity']:.2f}",
                )
                offs = sorted(
                    (h for h in rep["raw_hits"] if h["subject_id"] != _DEMO_EXCLUDE),
                    key=lambda h: -h["bitscore"],
                )[:10]
                st.dataframe(
                    [
                        {k: h[k] for k in ("subject_id", "bitscore", "pident", "qstart", "qend")}
                        for h in offs
                    ],
                    use_container_width=True,
                )
                st.info(
                    "These 'off-targets' are other IGHV alleles — including other alleles "
                    "of the same gene — because this conserved primer matches most of the "
                    "family. Great for a repertoire assay, poor for a gene-specific one."
                )

        st.divider()
        st.subheader("Check your own primers against any database")
        spec_primers = st.text_area(
            "Primer(s), one per line",
            height=90,
            key="spec_primers",
            placeholder="ATCCTCTTCTTGGTGGCAGC",
        )
        exclude = st.text_input(
            "Exclude on-target id (optional)", key="spec_exclude", help=_EXCLUDE_HINT
        )
        backend = st.radio(
            "Database source",
            ["Local BLAST DB", "Remote NCBI"],
            horizontal=True,
            key="spec_backend",
        )
        db_path, remote, database = None, False, "nt"
        if backend == "Local BLAST DB":
            db_path = st.text_input(
                "makeblastdb basename (any local DB you have)",
                value="",
                placeholder=_DEMO_DB,
                key="spec_localdb",
                help="e.g. the IGHV demo, or your own DB built with makeblastdb.",
            ).strip()
        else:  # Remote NCBI — pick any NCBI database by name
            remote = True
            database = st.text_input(
                "NCBI database name",
                value="nt",
                key="spec_ncbi",
                help="Any NCBI BLAST db, e.g. nt, refseq_rna, refseq_select_rna, human_genome.",
            ).strip()
            st.caption(
                "⏳ Remote BLAST is queued and rate-limited — expect ~30 s to several minutes."
            )

        if st.button("Run specificity check", key="spec_run"):
            primers = [s for s in (_clean_dna(p) for p in spec_primers.splitlines()) if s]
            if not primers:
                st.warning("Enter at least one primer.")
            elif not remote and not db_path:
                st.warning("Enter a local BLAST DB path.")
            else:
                rows = []
                with st.spinner("Running BLAST..."):
                    for p in primers:
                        try:
                            rep = check_specificity(
                                p,
                                db_path=db_path,
                                exclude_target_id=exclude.strip() or None,
                                remote=remote,
                                database=database,
                            )
                            rows.append(
                                {
                                    "primer": p,
                                    "source": rep["source"],
                                    "off_target_hits": rep["off_target_hit_count"],
                                    "best_bitscore": round(rep["best_off_target_bitscore"], 1),
                                    "3'_complementarity": round(
                                        rep["three_prime_offtarget_complementarity"], 2
                                    ),
                                }
                            )
                        except Exception as exc:  # noqa: BLE001
                            rows.append(
                                {"primer": p, "source": "error", "off_target_hits": str(exc)}
                            )
                st.dataframe(rows, use_container_width=True)
                st.caption(
                    "Remember: a low/0 count only means 'no hits in THIS database' — "
                    "trustworthy only if the DB represents the real off-target space."
                )


def main() -> None:
    """Entrypoint invoked by ``streamlit run copilot/app/main.py``."""
    render_app()


if __name__ == "__main__":
    main()
