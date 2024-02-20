# ML-InsightHub - Automated Machine Learning with Streamlit, Pandas Profiling, and PyCaret

 ML-Insight Hub is an application for Automated Machine Learning (AutoML) built using Streamlit, Pandas Profiling, and PyCaret. This application allows users to upload a dataset, perform Exploratory Data Analysis (EDA), build machine learning models for regression or classification, and download the trained models.

# Getting Started
- To run the application locally, make sure you have the required libraries installed. You can install them using:
```
pip install streamlit pandas ydata_profiling streamlit_pandas_profiling pycaret
```
- After installing the libraries, you can run the application by executing:
```
streamlit run app.py
```

# Application Structure
## Sidebar Navigation

- Upload: Upload your dataset for modeling.
- Profiling: Perform Automated Exploratory Data Analysis (EDA) on the dataset.
- ML Regression: Build automated machine learning models for regression.
- ML Classification: Build automated machine learning models for classification.
- Download: Download the trained machine learning model.

### Upload
- Upload your dataset using the file uploader.
- The uploaded data will be saved as "source_data.csv."

### Profiling
- Conduct Automated Exploratory Data Analysis (EDA) using Pandas Profiling.
- Gain insights into the dataset's structure, distribution, and statistics.

### ML Regression
- Train a machine learning model for regression.
- Select the target column and click "Train model."
- Compare different regression models and save the best one as "best_model_regression.pkl."

### ML Classification
- Train a machine learning model for classification.
- Select the target column and click "Train model."
- Compare different classification models and save the best one as "best_model_classification.pkl."

### Download
- Download the trained machine learning model based on the last chosen type (regression or classification).
- Use the provided code snippets to load and use the trained model in your own code.

## Usage Instructions
- Follow the sidebar navigation to perform specific tasks.
- After training a model, use the "Download" section to download the trained model file.
- Refer to the code snippets provided for using the trained model in your code.

## Note
- Ensure you have trained a model before attempting to download it.
- The application provides flexibility for regression tasks only. The application is still under development for classification tasks.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature_branch`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature_branch`.
5. Open a pull request.
