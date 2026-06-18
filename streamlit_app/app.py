from pathlib import Path
import streamlit as st

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Shilpa | AI/ML Portfolio",
    page_icon="✨",
    layout="wide"
)

# ------------------------------------
# Load Hero Image
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "hero_banner.png"

# ------------------------------------
# Hero Banner
# ------------------------------------
st.image(str(IMAGE_PATH), use_container_width=True)

# ------------------------------------
# Header
# ------------------------------------
st.title("Shilpa Sankar")

st.markdown("""
### Building AI systems that predict, personalize, and optimize.

Welcome to my AI/ML portfolio. Here you'll find projects focused on:

- 🎯 Customer Segmentation & Personalization
- 📉 Churn Prediction & Retention
- 📊 Customer Health Scoring
- 🛒 Recommendation Systems
- 📈 Forecasting & Demand Planning
- 💰 Price & Promotion Analytics
- 🤖 Applied Machine Learning
""")

st.divider()

# ------------------------------------
# Quick Stats
# ------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Projects", "10+")

with col2:
    st.metric("Domains", "6")

with col3:
    st.metric("Models", "ML & Forecasting")

with col4:
    st.metric("Focus", "Business AI")

st.divider()

# ------------------------------------
# Featured Projects
# ------------------------------------
st.subheader("⭐ Featured Projects")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **🎯 Customer Segmentation & Personalization**

    Identify customer groups and tailor experiences using clustering techniques.
    """)

    st.info("""
    **📉 Churn Retention Engine**

    Predict customer churn and recommend retention actions.
    """)

    st.info("""
    **📊 Customer Health Score**

    Measure customer engagement and identify at-risk accounts.
    """)

with col2:
    st.success("""
    **🛒 Airport Recommendation System**

    Recommend airport services and experiences based on user preferences.
    """)

    st.success("""
    **📈 Price Volatility Forecasting**

    Forecast future price movements using machine learning models.
    """)

    st.success("""
    **♻️ Waste Prediction**

    Predict inventory waste and improve planning decisions.
    """)

st.divider()

# ------------------------------------
# Tech Stack
# ------------------------------------
st.subheader("🛠️ Tech Stack")

st.markdown("""
**Languages & Libraries**

`Python` • `Pandas` • `NumPy` • `Scikit-Learn`

**Visualization & Apps**

`Streamlit` • `Plotly`

**Machine Learning**

`Classification` • `Clustering` • `Forecasting`

**Business Analytics**

`Customer Analytics` • `Recommendation Systems`
""")

st.divider()

# ------------------------------------
# Footer
# ------------------------------------
st.subheader("🔗 Connect")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        "[GitHub](https://github.com/shilpasankar)"
    )

with c2:
    st.markdown(
        "[LinkedIn](https://linkedin.com)"
    )

st.caption("Built with Streamlit ❤️")
