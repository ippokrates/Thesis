"""
app/shared.py — UI helpers shared by both tabs
================================================
`model_consensus`, `render_prob_metrics`, and `THESIS_DIR` live here so the
two tab modules do not duplicate each other or import from each other.
This module draws (Streamlit calls) but runs nothing on import.
"""

from pathlib import Path

import streamlit as st


THESIS_DIR = Path(__file__).parent.parent.resolve()


def model_consensus(prob_a: float, prob_b: float, name_a: str, name_b: str,
                    positive: str, negative: str) -> None:
    """Green (agree) / Orange (disagreement) consensus badge shared by both tabs."""
    agree = (prob_a > 0.5) == (prob_b > 0.5)
    if agree:
        label = positive if prob_a > 0.5 else negative
        st.success(f"**Σύγκλιση / Consensus** — {name_a} & {name_b} συμφωνούν: **{label}**")
    else:
        st.warning(
            f"**Διαφωνία / Disagreement** — {name_a} & {name_b} διαφωνούν.\n\n"
            "Συνιστάται κλινικός έλεγχος / Clinical review recommended."
        )


def render_prob_metrics(cards: list[tuple[str, float, str, str, str]]) -> None:
    """Render one row of metric cards.

    cards: list of (label, prob, positive, negative, delta_prefix).
    """
    cols = st.columns(len(cards))
    for col, (label, prob, positive, negative, prefix) in zip(cols, cards):
        with col:
            st.metric(
                label=label,
                value=f"🔴 {positive}" if prob > 0.5 else f"🟢 {negative}",
                delta=f"{prefix} = {prob * 100:.1f}%",
            )
