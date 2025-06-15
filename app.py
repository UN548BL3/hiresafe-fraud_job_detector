import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from wordcloud import WordCloud
import pycountry

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="HireSafe - Job Fraud Detector", layout="wide")
st.title("🔍 HireSafe - Job Fraud Detection System")
st.markdown("Upload a CSV of job postings to detect **fraudulent** entries using machine learning.")

# ----------------- Load Model with caching -----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("fraud_detector.pkl")

model = load_model()

# ----------------- Upload File -----------------
uploaded_file = st.file_uploader("📤 Upload a job postings CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ----------------- Validate Columns -----------------
    required_columns = [
        'title', 'description', 'requirements', 'company_profile', 'benefits',
        'location', 'employment_type', 'required_experience',
        'required_education', 'industry', 'function', 'salary_range'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # ----------------- Preprocessing -----------------
    for col in ['title', 'description', 'requirements', 'company_profile', 'benefits']:
        df[col] = df[col].fillna('')
    df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements']
    df['desc_len'] = df['description'].apply(len)
    df['company_profile_len'] = df['company_profile'].apply(len)
    df['benefits_len'] = df['benefits'].apply(len)
    df['has_salary'] = df['salary_range'].notnull().astype(int)

    def parse_salary(s):
        try:
            s = str(s).replace('$', '').replace(',', '')
            nums = [float(n.strip()) for n in s.split('-') if n.strip().replace('.', '').isdigit()]
            return sum(nums) / len(nums) if nums else 0.0
        except:
            return 0.0

    df['salary_avg'] = df['salary_range'].apply(parse_salary)

    input_df = df[['text', 'location', 'employment_type', 'required_experience',
                'required_education', 'industry', 'function',
                'desc_len', 'company_profile_len', 'benefits_len', 'has_salary', 'salary_avg']]

    # ----------------- Predictions -----------------
    with st.spinner('Running fraud detection...'):
        df['fraud_prob'] = model.predict_proba(input_df)[:, 1]
        df['fraud_prediction'] = model.predict(input_df)
        df['label'] = df['fraud_prediction'].map({0: 'Genuine', 1: 'Fraudulent'})

    # ----------------- SHAP Explainability -----------------
    X_preprocessed = model.named_steps['preprocessor'].transform(input_df)
    if hasattr(X_preprocessed, "toarray"):
        X_preprocessed = X_preprocessed.toarray()

    classifier = model.named_steps['classifier']

    @st.cache_data(show_spinner=False)
    def get_shap_explainer(_clf, _data):
        return shap.LinearExplainer(_clf, _data, feature_perturbation="interventional")

    explainer = get_shap_explainer(classifier, X_preprocessed)

    st.subheader("🔎 SHAP Explainability – Why was this job flagged?")

    try:
        selected_job = st.selectbox("📝 Choose a job to explain", df['title'].unique())
        selected_index = df[df['title'] == selected_job].index[0]
        sample_preprocessed = X_preprocessed[[selected_index]]
        shap_values = explainer(sample_preprocessed)

        st.markdown(f"**Job Title:** {selected_job}")
        st.markdown(f"**Prediction:** {df.loc[selected_index, 'label']}")
        st.markdown(f"**Fraud Probability:** {df.loc[selected_index, 'fraud_prob']:.2f}")
        st.write("**Feature Contribution Breakdown:**")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning("⚠️ SHAP explanation not available.")
        st.text(str(e))

    # ----------------- Download Predictions -----------------
    st.download_button("⬇️ Download predictions", df.to_csv(index=False), "fraud_predictions.csv", "text/csv")

    # ----------------- Insights & Analytics -----------------
    st.subheader("📊 Insights & Fraud Detection Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🔍 Total Jobs Analyzed", len(df))
        st.metric("⚠️ Detected Fraudulent Jobs", int((df['fraud_prediction'] == 1).sum()))
        fig = px.histogram(df, x="fraud_prob", nbins=20, color="label", title="Fraud Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names="label", title="Fraud vs Genuine")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------- Location-wise Heatmap -----------------
    st.markdown("### 🌍 Fraud Distribution by Location")
    location_df = df.groupby('location')['fraud_prediction'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(location_df, x=location_df.index, y=location_df.values,
                labels={'x': 'Location', 'y': 'Avg Fraud Probability'})
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- Global Choropleth -----------------
    df['country'] = df['location'].str.extract(r',\s*([\w\s]+)$')

    def get_country_code(name):
        try:
            country = pycountry.countries.lookup(name)
            return country.alpha_3
        except:
            return None

    df['country_code'] = df['country'].apply(lambda x: get_country_code(x) if pd.notnull(x) else None)
    choropleth_df = df.dropna(subset=['country_code'])

    fig = px.choropleth(
        choropleth_df,
        locations="country_code",
        color="fraud_prediction",
        hover_name="country",
        title="🌍 Global Fraud Distribution",
        color_continuous_scale=px.colors.sequential.Reds
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- Rule-Based Red Flags -----------------
    def has_red_flags(text):
        red_words = ["quick money", "work from home", "immediate start", "click here"]
        return int(any(word in text.lower() for word in red_words))

    df['red_flag'] = df['text'].apply(has_red_flags)

    st.markdown("### 🚩 Jobs with Red-Flag Keywords")
    st.dataframe(df[df['red_flag'] == 1][['title', 'location', 'fraud_prob']])

    st.markdown("### 🔎 Top Suspicious Job Titles")
    top_jobs = df[df['fraud_prediction'] == 1].sort_values(by='fraud_prob', ascending=False).head(10)
    st.dataframe(top_jobs[['title', 'location', 'salary_range', 'fraud_prob']])

    st.subheader("☁️ Common Words in Fraudulent Listings")
    fraud_text = " ".join(df[df['fraud_prediction'] == 1]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fraud_text)
    st.image(wordcloud.to_array())

    st.markdown("### 🎛 Filter Predictions by Threshold")
    prob_threshold = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.01)
    filtered_df = df[df['fraud_prob'] >= prob_threshold].sort_values(by='fraud_prob', ascending=False)
    st.write(f"Showing {len(filtered_df)} jobs with fraud probability ≥ {prob_threshold}")
    st.dataframe(filtered_df[['title', 'location', 'fraud_prob', 'label']])

    st.success("✅ Dashboard updated with interactive analytics.")

else:
    st.info("📥 Please upload a CSV file to begin.")

# ----------------- Optional UI Cleanup -----------------
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



