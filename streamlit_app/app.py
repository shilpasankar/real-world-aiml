import streamlit as st

st.set_page_config(
    page_title="Shilpa | AI/ML Portfolio",
    page_icon="✨",
    layout="wide"
)

# Hero image
st.image(
    "assets/hero_banner.png",
    use_container_width=True
)

# Title
st.title("Shilpa Sankar")

st.markdown("""
### Building AI systems that predict, personalize, and optimize.

Explore projects spanning:

- 🎯 Customer Segmentation
- 📉 Churn Prediction
- 🛒 Recommendation Systems
- 📈 Forecasting & Demand Planning
- 💰 Price & Promotion Analytics
- 🤖 Applied Machine Learning
""")

st.divider()

# Portfolio Highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Projects", "10+")

with col2:
    st.metric("Domains", "6")

with col3:
    st.metric("ML Solutions", "Business-Focused")

st.divider()

st.subheader("Featured Projects")

st.markdown("""
⭐ **Customer Segmentation & Personalization**

⭐ **Churn Retention Engine**

⭐ **Customer Health Score**

⭐ **Price Volatility Forecasting**

⭐ **Airport Recommendation System**

⭐ **Waste Prediction**
""")
