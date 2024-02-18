# Import necessary libraries
import streamlit as st
import pandas as pd
import os

# Import profiling libraries
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Import ML page libraries
from pycaret.regression import setup, compare_models, create_model, tune_model, pull, save_model, load_model, predict_model

# Set up Streamlit sidebar
with st.sidebar:
    # Add an image and title to the sidebar
    st.image("https://img.freepik.com/free-vector/hand-drawn-flat-design-npl-illustration_23-2149277640.jpg?w=740&t=st=1708236522~exp=1708237122~hmac=c75ab7bb3b9088060d1b2c2a02176edf10c21169bb32ce4bd4ada27603c94aca", width=200, caption="AutoML", use_column_width=True)
    st.title("ExploreML")
    st.write("Automated Machine Learning")

    # Add navigation options to the sidebar
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML Regression", "ML Classification", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret." )
    st.info("You can upload a dataset, perform EDA, build a model, and download the results.")

# Check if the dataset exists and load it
if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col=None)

# Handle different user choices
if choice == "Upload":
    # Upload data option
    st.title("Upload Your Data for Modeling!")
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("source_data.csv", index=None)
        st.dataframe(df)
        st.success("Data has been uploaded successfully.")

elif choice == "Profiling":
    # EDA option
    st.title("Automated Exploratory Data Analysis (EDA)")
    profile_report_df = ProfileReport(df)
    st_profile_report(profile_report_df)

elif choice == "ML Regression":
    # ML Regression option
    last_model_type = 'regression'
    st.title("Automated Machine Learning - Regression")
    target = st.selectbox("Select your target column", df.columns)
    
    # Train model button
    if st.button("Train model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML experiment settings.")
        st.dataframe(setup_df)
        
        # Compare models and select the best one
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        st.info("This is the best model")
        best_model
        
        # Save the best model
        save_model(best_model, "best_model_regression")
        st.info("The best model has been saved as best_model_regression.pkl")

    # Save the model type as a text file
    with open('model_type.txt', 'w') as f:
        f.write('regression')

elif choice == "ML Classification":
    # ML Classification option
    last_model_type = 'classification'
    st.title("Automated Machine Learning - Classification")
    target = st.selectbox("Select your target column", df.columns)

    # Train model button
    if st.button("Train model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML experiment settings.")
        st.dataframe(setup_df)

        # Compare models and select the best one
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        st.info("This is the best model")
        best_model

        # Save the best model
        save_model(best_model, "best_model_classification")
        st.info("The best model has been saved as best_model_classification.pkl")

    # Save the model type as a text file
    with open('model_type.txt', 'w') as f:
        f.write('classification')

elif choice == "Download":
    # Download option
    if os.path.exists("model_type.txt"):
        with open("model_type.txt", 'r') as f:
            last_model_type = f.read().strip()
    else:
        last_model_type = None

    # Check the model type and provide download button accordingly
    if last_model_type == 'regression':
        with open("best_model_regression.pkl", 'rb') as f:
            st.markdown("### Download the trained model file (Only for Regression Problem) ###")
            st.download_button("Download", f, "trained_model_regression.pkl")
    elif last_model_type == 'classification':
        with open("best_model_classification.pkl", 'rb') as f:
            st.markdown("### Download the trained model file (Only for Classification Problem) ###")
            st.download_button("Download", f, "trained_model_classification.pkl")
    else:
        st.warning("Please train a model first.")

    # Provide instructions for using the trained model in code
    st.markdown("""
    Note: If you want to use the trained model in your code use the below code for it:
    ```python
    import pickle
    with open("trained_model_regression.pkl", 'rb') as f:
        model = pickle.load(f)
        model.predict(X)
    ```

    or 

    ```python
    from pycaret.regression import load_model
    model = load_model("trained_model_regression")
    model.predict(X)
    ```

    or 

    ```python
    from pycaret.classification import load_model
    model = load_model("trained_model_classification")
    model.predict(X)
    ```

    Note this may change based on the model you have trained.
    """)

# Add a footer
st.markdown("""
---
**Copyright © 2024 Dhrumit Patel. All rights reserved.**
""")
