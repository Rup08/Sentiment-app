import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

# Set page layout to wide for better graph viewing
st.set_page_config(page_title="Sentiment Analyzer Pro", layout="wide")

# Create a Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Predict New Data", "2. Model Performance Metrics"])

# Load the PyCaret pipeline
@st.cache_resource
def get_model():
    # Make sure this matches your .pkl file name on GitHub (without the .pkl extension)
    return load_model("my_sentiment_pipeline") 

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading the model. Make sure 'my_sentiment_pipeline.pkl' is in your repository. Error details: {e}")
    st.stop()

# ==========================================
# PAGE 1: PREDICT NEW DATA
# ==========================================
if page == "1. Predict New Data":
    st.title("📊 Live Sentiment Prediction")
    st.write("Upload a CSV file to get instant sentiment predictions. The file must contain a **'review_text'** column.")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head(3))
        
        if 'review_text' not in df.columns:
            st.error("Error: The uploaded CSV must contain a column specifically named 'review_text'.")
        else:
            if st.button("Run Predictions"):
                with st.spinner("Analyzing text and predicting sentiment..."):
                    try:
                        # 1. Recreate the feature engineering from your training code
                        # Your training code created 'review_length', so the model needs it to run!
                        if 'review_length' not in df.columns:
                            df['review_length'] = df['review_text'].astype(str).apply(len)
                        
                        # 2. Predict using PyCaret
                        predictions_df = predict_model(model, data=df)
                        
                        st.success("Analysis Complete!")
                        st.write("### Prediction Results:")
                        st.dataframe(predictions_df)
                        
                        # 3. Allow downloading
                        csv = predictions_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.info("Tip: Make sure your uploaded CSV has all the columns the model was trained on (like 'rating' if it was included in the training dataset).")

# ==========================================
# PAGE 2: MODEL PERFORMANCE METRICS
# ==========================================
elif page == "2. Model Performance Metrics":
    st.title("📈 Model Evaluation Dashboard")
    st.write("These charts were generated automatically during the PyCaret training and hyperparameter tuning phase.")
    
    # PyCaret saves files with specific default names. 
    # We create a dictionary of the files your code generated.
    expected_plots = {
        'ROC Curve': 'AUC.png',
        'Confusion Matrix': 'Confusion Matrix.png',
        'Precision-Recall': 'Precision Recall.png',
        'Feature Importance': 'Feature Importance.png',
        'Learning Curve': 'Learning Curve.png',
        'Decision Boundary': 'Decision Boundary.png',
        'Prediction Error': 'Prediction Error.png',
        'Class Report': 'Class Report.png'
    }
    
    # Create two columns to display images side-by-side
    col1, col2 = st.columns(2)
    images_found = False
    
    # Loop through the expected plots and display them if they exist in the GitHub repo
    for i, (title, filename) in enumerate(expected_plots.items()):
        if os.path.exists(filename):
            images_found = True
            # Alternate placing images in col1 and col2
            if i % 2 == 0:
                with col1:
                    st.image(filename, caption=title, use_column_width=True)
            else:
                with col2:
                    st.image(filename, caption=title, use_column_width=True)
                    
    if not images_found:
        st.warning("⚠️ No graph images were found.")
        st.info("To see your graphs here, upload the `.png` files (like 'AUC.png' and 'Confusion Matrix.png') from your computer directly into your GitHub repository!")