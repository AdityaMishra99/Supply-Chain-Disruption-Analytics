#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import datetime
import io
from PIL import Image

# ----------------------------------
# Professional Header Image
# ----------------------------------
st.image(
    "https://logowik.com/content/uploads/images/maersk-line8921.logowik.com.webp",
    use_container_width=True
)

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("📊 Dashboard Controls")
st.sidebar.info("Executive dashboard for supply chain risk assessment")

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("supply_chain_recovery_model.pkl")

model = load_model()

# ----------------------------------
# PDF Report Generator
# ----------------------------------
def generate_pdf_report(input_df, prediction, risk_level, importance_df):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Supply Chain Recovery Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Input Summary</b>", styles["Heading2"]))
    for col, val in input_df.iloc[0].items():
        elements.append(Paragraph(f"{col}: {val}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Estimated Recovery Time: <b>{round(prediction)} days</b>", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: <b>{risk_level}</b>", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Top Feature Importance</b>", styles["Heading2"]))

    table_data = [["Feature", "Importance"]]
    for _, row in importance_df.head(10).iterrows():
        table_data.append([row["Feature"], round(row["Importance"], 4)])

    table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------------------------------
# App Title
# ----------------------------------
st.title("Supply Chain Recovery Prediction")
st.markdown("Predict **full recovery time (days)** after a supply chain disruption.")
st.divider()

# ----------------------------------
# Input Form
# ----------------------------------
with st.form("prediction_form"):

    st.subheader("📊 Operational Inputs")
    supplier_tier = st.selectbox("Supplier Tier", [1, 2, 3, 4])
    disruption_severity = st.slider("Disruption Severity (1–5)", 1, 5, 3)
    production_impact_pct = st.slider("Production Impact (%)", 0.0, 100.0, 25.0)
    response_time_days = st.number_input("Response Time (Days)", min_value=0, value=7)
    revenue_loss = st.number_input("Revenue Loss (USD)", min_value=0, value=100000)

    st.subheader("🏷️ Contextual Inputs")
    disruption_type = st.selectbox("Disruption Type", ["Logistics Delay", "Natural Disaster", "Supplier Bankruptcy", "Labor Strike"])
    industry = st.selectbox("Industry", ["Manufacturing", "Automotive", "Pharmaceutical", "Retail"])
    supplier_region = st.selectbox("Supplier Region", ["Asia", "Europe", "North America", "South America"])
    supplier_size = st.selectbox("Supplier Size", ["Small", "Medium", "Large"])

    submit = st.form_submit_button("🔮 Predict Recovery Time")

# ----------------------------------
# Prediction + Dashboard
# ----------------------------------
if submit:

    # ---- Input dataframe ----
    input_data = pd.DataFrame([{
        "supplier_tier": supplier_tier,
        "disruption_severity": disruption_severity,
        "production_impact_pct": production_impact_pct,
        "response_time_days": response_time_days,
        "log_revenue_loss": np.log1p(revenue_loss),
        "disruption_type": disruption_type,
        "industry": industry,
        "supplier_region": supplier_region,
        "supplier_size": supplier_size
    }])

    # ---- Prediction ----
    prediction = model.predict(input_data)[0]

    # ----------------------------------
    # Risk Classification + Visual
    # ----------------------------------
    st.subheader("Risk Visual Indicator")

    if prediction <= 10:
        risk_level = "Low"
        st.image(
            "https://static.vecteezy.com/system/resources/previews/013/916/245/original/low-risk-icon-on-white-background-illustration-eps-10-vector.jpg",
            use_container_width=True
        )
    elif prediction <= 30:
        risk_level = "Medium"
        st.image(
            "https://static.vecteezy.com/system/resources/previews/002/191/777/large_2x/risk-icon-on-speedometer-medium-risk-meter-isolated-on-white-background-vector.jpg",
            use_container_width=True
        )
    else:
        risk_level = "High"
        st.image(
            "https://static.vecteezy.com/system/resources/previews/013/916/244/original/high-risk-icon-on-white-background-illustration-eps-10-vector.jpg",
            use_container_width=True
        )

    # ----------------------------------
    # KPI Metrics
    # ----------------------------------
    st.divider()
    st.subheader("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recovery Time (Days)", round(prediction))
    col2.metric("Production Impact", f"{production_impact_pct}%")
    col3.metric("Revenue Loss", f"${revenue_loss:,}")
    col4.metric("Disruption Severity", f"{disruption_severity}/5")

    # ----------------------------------
    # Feature Importance
    # ----------------------------------
    rf_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    importance_df = pd.DataFrame({
        "Feature": preprocessor.get_feature_names_out(),
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"][:15], importance_df["Importance"][:15])
    ax.invert_yaxis()
    st.pyplot(fig)

    # ----------------------------------
    # PDF Download
    # ----------------------------------
    pdf_buffer = generate_pdf_report(
        input_data,
        prediction,
        risk_level,
        importance_df
    )

    st.download_button(
        "Download Executive PDF Report",
        data=pdf_buffer,
        file_name="supply_chain_recovery_report.pdf",
        mime="application/pdf"
    )

    st.caption("Prediction powered by RandomForest + preprocessing pipeline")


# In[ ]:




