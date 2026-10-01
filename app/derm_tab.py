"""
app/derm_tab.py — Dermatology tab
==================================
Skin lesion analysis: uploads an image, calls diagnose.py as a subprocess
(CNN MobileNetV2 + Grad-CAM, ViT-B/16 + Attention Rollout) and shows the
composite result plus the two probabilities and their consensus.

Imported by app.py, which is the Streamlit entry point. This module only
draws, it does not run anything on import.
"""

import os
import json
import sys
import subprocess
import tempfile
from pathlib import Path

import streamlit as st


THESIS_DIR    = Path(__file__).parent.parent.resolve()
DIAGNOSE_PY   = THESIS_DIR / "diagnose.py"


def run_diagnose(image_path: str, alpha: float) -> dict:
    """
    Invoke diagnose.py as a subprocess (clean env — no TF_USE_LEGACY_KERAS).
    Runs with --json, so the result is read from the last stdout line instead
    of scraping the human-readable results block.
    Returns {"cnn_prob": float, "vit_prob": float, "output_png": str | None}.
    """
    clean_env = {k: v for k, v in os.environ.items() if k != "TF_USE_LEGACY_KERAS"}
    clean_env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    proc = subprocess.run(
        [
            sys.executable, str(DIAGNOSE_PY),
            "--image", image_path,
            "--alpha", str(alpha),
            "--json",
        ],
        env=clean_env,
        capture_output=True,
        text=True,
        cwd=str(THESIS_DIR),
    )

    if proc.returncode != 0:
        detail = (proc.stderr or "")
        if proc.stdout:
            detail += ("\n" if detail else "") + proc.stdout
        raise RuntimeError(detail or "(empty output from diagnose.py)")

    stdout = proc.stdout
    # diagnose.py prints the JSON object as its last line. Take the last
    # non-empty line because TensorFlow may emit warnings on stdout before it.
    json_line = [l for l in stdout.strip().splitlines() if l.strip()][-1]
    data = json.loads(json_line)

    return {
        "cnn_prob":   data["cnn_prob"],
        "vit_prob":   data["vit_prob"],
        "output_png": data["output_path"],
        "stdout":     stdout,
    }


def derm_consensus(cnn_prob: float, vit_prob: float) -> None:
    """Green (agree) / Orange (disagree) consensus badge for skin lesion tab."""
    agree = (cnn_prob > 0.5) == (vit_prob > 0.5)
    if agree:
        label = "Κακοήθης / Malignant" if cnn_prob > 0.5 else "Καλοήθης / Benign"
        st.success(f"**Σύγκλιση / Consensus** — CNN & ViT συμφωνούν: **{label}**")
    else:
        st.warning(
            "**Διαφωνία / Disagreement** — CNN & ViT διαφωνούν.\n\n"
            "Συνιστάται κλινικός έλεγχος / Clinical review recommended."
        )


def render_dermatology_tab() -> None:
    st.header("Ανάλυση Δερματικής Βλάβης / Skin Lesion Analysis")
    # st.markdown(
    #     "Ανεβάστε μία εικόνα δερματικής βλάβης. Το σύστημα εκτελεί πρόβλεψη με "
    #     "**CNN (MobileNetV2)** και **ViT-B/16** και παράγει XAI επεξηγήσεις με "
    #     "**Grad-CAM** και **Attention Rollout**.\n\n"
    #     "*Upload a skin lesion image. The system predicts with CNN and ViT and generates "
    #     "XAI explanations via Grad-CAM and Attention Rollout.*"
    # )
    #st.divider()

    # Upload & settings
    col_up, col_cfg = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Εικόνα δέρματος / Skin image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="JPEG, PNG ή BMP",
        )

    with col_cfg:
        alpha = st.slider(
            "Διαφάνεια heatmap / Heatmap opacity",
            min_value=0.1, max_value=0.9, value=0.4, step=0.05,
            help="Πόσο εμφανές να είναι το XAI overlay / Strength of the XAI overlay",
        )

    if uploaded is None:
        st.info(
            "Ανεβάστε μία εικόνα για να ξεκινήσει η ανάλυση.\n\n"
            "*Upload an image to start the analysis.*"
        )
        return

    # Preview of uploaded image
    st.image(uploaded, caption=f" **{uploaded.name}**", width=280)

    if not st.button("Ανάλυση / Analyze", type="primary", width='stretch'):
        return

    # Inference via subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix   = Path(uploaded.name).suffix or ".jpg"
        img_path = os.path.join(tmpdir, f"input{suffix}")

        with open(img_path, "wb") as f:
            f.write(uploaded.getvalue())

        with st.spinner(
            " *Ανάλυση εικόνας και παραγωγή επεξηγηματικών χαρτών*/*Image analysis and production of explanatory maps*"
        ):
            try:
                result = run_diagnose(img_path, alpha)
            except RuntimeError as err:
                st.error(f"Σφάλμα / Error:\n```\n{err}\n```")
                return

    cnn_prob   = result["cnn_prob"]
    vit_prob   = result["vit_prob"]
    output_png = result["output_png"]
    run_stdout = result.get("stdout", "")

    png_bytes = None
    if output_png and os.path.exists(output_png):
        with open(output_png, "rb") as f:
            png_bytes = f.read()

    if cnn_prob is not None and vit_prob is not None:
        st.divider()
        st.subheader("Πιθανότητες / Probabilities")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric(
                label="CNN — MobileNetV2",
                value="🔴 Κακοήθης / Malignant" if cnn_prob > 0.5 else "🟢 Καλοήθης / Benign",
                delta=f"P(malignant) = {cnn_prob * 100:.1f}%",
            )
        with mc2:
            st.metric(
                label="ViT — ViT-B/16",
                value="🔴 Κακοήθης / Malignant" if vit_prob > 0.5 else "🟢 Καλοήθης / Benign",
                delta=f"P(malignant) = {vit_prob * 100:.1f}%",
            )
        with mc3:
            avg = (cnn_prob + vit_prob) / 2.0
            st.metric(
                label="Ensemble — Μέσος Όρος / Average",
                value="🔴 Κακοήθης / Malignant" if avg > 0.5 else "🟢 Καλοήθης / Benign",
                delta=f"P(malignant) = {avg * 100:.1f}%",
            )

        derm_consensus(cnn_prob, vit_prob)

    if png_bytes:
        st.divider()
        st.subheader("Αποτελέσματα XAI / XAI Results")
        st.image(png_bytes, width='stretch')
    else:
        st.warning("Η διάγνωση απέτυχε / Diagnosis failed.")
        if run_stdout.strip():
            with st.expander("Λεπτομέρειες / Details"):
                st.code(run_stdout)

    # ── XAI note ─────────────────────────────────────────────────────────────
    # with st.expander("ℹΕπεξήγηση μεθόδων XAI / About XAI methods"):
    #     st.markdown(
    #         "**Grad-CAM** (CNN): Χρησιμοποιεί τις κλίσεις του τελευταίου convolutional layer "
    #         "για να αναδείξει τις περιοχές που επηρέασαν την πρόβλεψη.\n\n"
    #         "*Uses gradients of the last convolutional layer to highlight regions influencing the prediction.*\n\n"
    #         "**Attention Rollout** (ViT): Διαδίδει τα attention weights σε όλα τα 12 transformer layers "
    #         "για να παραχθεί ένας χάρτης εστίασης.\n\n"
    #         "*Propagates attention weights across all 12 transformer layers to produce a focus map.*"
    #     )

    if png_bytes:
        st.download_button(
            label="Λήψη Αποτελέσματος / Download Result (PNG)",
            data=png_bytes,
            file_name=f"xai_skin_{Path(uploaded.name).stem}.png",
            mime="image/png",
            width='stretch',
        )
