import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import missingno as msno

# 📄 Configure page
st.set_page_config(page_title="EDA App", layout="wide")
st.title("📊 Automated EDA Web App")

# 📤 Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # 📋 Dataset Overview
    st.subheader("📋 Dataset Overview")
    st.write(df.head())

    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("**Column Types**")
    st.dataframe(df.dtypes.astype(str))

    # 📈 Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all').T)

    # 📉 Missing Values Heatmap
    st.subheader("📉 Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    msno.heatmap(df, ax=ax)
    st.pyplot(fig)

    # 📊 Univariate Analysis
    st.subheader("📊 Univariate Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns found.")
    if not categorical_cols:
        st.warning("⚠️ No categorical columns found.")

    col1, col2 = st.columns(2)

    with col1:
        if numeric_cols:
            num_col = st.selectbox("Select a numeric column", numeric_cols)
            fig = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if categorical_cols:
            cat_col = st.selectbox("Select a categorical column", categorical_cols)
            cat_data = df[cat_col].value_counts().reset_index()
            cat_data.columns = [cat_col, "count"]
            fig = px.bar(cat_data, x=cat_col, y="count", title=f"Count of {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    # 🔗 Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # 🔍 Pairplot
    st.subheader("🔍 Pairplot")
    if len(numeric_cols) > 1:
        selected = st.multiselect("Select numeric columns for pairplot (3 max)", numeric_cols, default=numeric_cols[:3])
        if selected and len(selected) <= 3:
            fig = sns.pairplot(df[selected])
            st.pyplot(fig)
        elif len(selected) > 3:
            st.warning("⚠️ Please select up to 3 columns for pairplot.")

    # 📦 Boxplot
    st.subheader("📦 Boxplot")
    col3, col4 = st.columns(2)
    with col3:
        y_col = st.selectbox("Y-axis (numeric)", numeric_cols)
    with col4:
        x_col = st.selectbox("X-axis (categorical)", categorical_cols)

    if x_col and y_col:
        fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.success("✅ EDA Completed!")

else:
    st.info("📤 Please upload a CSV file to begin.")
