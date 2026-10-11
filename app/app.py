import os

# Silence TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import streamlit as st

from derm_tab import render_dermatology_tab
from cardio_tab import render_cardiology_tab


st.set_page_config(
    page_title="XAI - Πτυχιακή",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* Slightly larger section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #a0aec0;
    margin-top: 0.5rem;
    margin-bottom: 0.3rem;
}
/* Tone down metric delta red/green — keep neutral grey */
[data-testid="stMetricDelta"] { color: #718096 !important; }
</style>
""", unsafe_allow_html=True)


def main() -> None:
    st.title("XAI Σύστημα Υποστήριξης Κλινικών Αποφάσεων")
    st.caption(
        "XAI Health Decision Support System  ·  "
        "Πτυχιακή: *«Ανάπτυξη XAI εργαλείου για ερμηνεία αποφάσεων σε συστήματα υγείας»*"
    )

    tab_derm, tab_cardio = st.tabs([
        "🔬 Δερματολογία / Dermatology",
        "🫀 Καρδιολογία / Cardiology",
    ])

    with tab_derm:
        render_dermatology_tab()

    with tab_cardio:
        render_cardiology_tab()


if __name__ == "__main__":
    main()
