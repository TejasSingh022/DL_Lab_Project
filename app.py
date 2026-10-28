"""Run with: streamlit run app.py"""

import streamlit as st
import torch

from src.model_loader import load_model
from src.predict import predict

st.set_page_config(page_title="ICD-9 Code Prediction", layout="centered")
st.title("ICD-9 Code Prediction")
st.write("Paste a discharge summary to see the top 10 PLM-ICD predictions.")
st.caption("Input is limited to 3,072 tokens. Scores are independent sigmoid outputs, not calibrated confidence percentages.")


@st.cache_resource(show_spinner="Loading PLM-ICD…")
def get_predictor():
    return load_model()


with st.form("prediction"):
    text = st.text_area("Discharge summary", height=280)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    if not text.strip():
        st.warning("Enter a discharge summary before predicting.")
    else:
        try:
            predictor = get_predictor()
            with st.spinner("Predicting ICD-9 codes…"):
                rows = predict(text, predictor)
            st.caption(f"Device: {predictor.device}")
            st.dataframe(
                rows, hide_index=True, use_container_width=True,
                column_config={"Prediction score": st.column_config.NumberColumn(format="%.6f")},
            )
            if any(row["Description"] == "Description unavailable" for row in rows):
                st.info("Some descriptions are absent from the supplied mapping; their predicted codes and scores are unchanged.")
        except torch.cuda.OutOfMemoryError:
            st.error("GPU memory is insufficient. Close GPU applications or restart with CUDA_VISIBLE_DEVICES=-1 to use CPU.")
        except Exception as exc:
            st.error(f"Prediction could not run: {exc}")
