import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("📊 Batch Sentiment Analysis App")
st.write("Upload a CSV file of text reviews to instantly predict their sentiment.")

# Load the PyCaret pipeline
@st.cache_resource
def get_model():
    return load_model("my_sentiment_pipeline") 

model = get_model()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Preview of your data:")
    st.dataframe(df.head(3))
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing sentiment using PyCaret..."):
            try:
                # Predict using PyCaret
                predictions_df = predict_model(model, data=df)
                
                st.success("Analysis Complete!")
                st.write("### Results:")
                st.dataframe(predictions_df)
                
                # Download button
                csv = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="pycaret_sentiment_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")