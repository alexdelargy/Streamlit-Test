import streamlit as st 
import pandas as pd 


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import time

st.set_page_config(page_title="Capstone Project", layout="wide")
st.title("Capstone Project")

st.sidebar.header("Step 1: Upload File", divider=True)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    st.dataframe(df.head())

    st.sidebar.header("Step 2: Select Features and Target", divider=True)
    features = []
    st.sidebar.write("Select Features")
    for column in df.columns:
        if st.sidebar.checkbox(column):
            features.append(column)

    target = st.sidebar.selectbox("Select Target", set(df.columns) - set(features))

    st.sidebar.header("Step 3: Train-Test Split", divider=True)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.9, 0.2, 0.1)
    random_state = st.sidebar.slider("Random State", 0, 100, 50, 1)

    st.sidebar.head("Step 4: Model Training", divider=True)
    if st.sidebar.button("Train Model"):
        with st.spinner("Training Model...", show_time=True):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.success("Model Trained Successfully!")

            time.sleep(1)
            
            X_test = scaler.transform(X_test)
            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            st.success(f"Model Evaluation Completed! RMSE: {rmse:.2f}")
