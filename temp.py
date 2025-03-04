import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from dotenv import load_dotenv
from crewai import Crew, Process
from crewai import LLM
from Agent.agents import DiseasePredictorAgent
from Agent.tasks import DiseasePredictorTask

# Load environment variables
load_dotenv()
GEMINI_API_KEY = "AIzaSyAoXkCAtHMMxphFRSHP1Wc78asPiY-pYC4"  # Replace with your Gemini API key

# Load the trained model
MODEL_PATH = "C:\\Users\\Vaibhavi\\Desktop\\cavscann-main\\Model_Assets\\dental_v1_vgg16.h5"

@st.cache_resource
def load_trained_model():
    """Load the trained TensorFlow model."""
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# Define class labels
class_labels = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth ulcer', 'Tooth Discoloration']

# Function to classify uploaded image
def classify_image(image,model):
    """Classify the uploaded image using the trained model."""
    img = image.resize((256, 256))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, predictions[0]

# CrewAI-powered Disease Information Fetcher
def DiseasePredictor(llm, disease):
    """Fetch disease information using CrewAI."""
    agent = DiseasePredictorAgent()
    task = DiseasePredictorTask()

    researcher_agent = agent.create_disease_researcher(llm, disease, class_labels)
    researcher_task = task.create_disease_research_task(researcher_agent, disease, class_labels)

    crew = Crew(
        agents=[researcher_agent],  
        tasks=[researcher_task],   
        process=Process.sequential,  
    )

    result = crew.kickoff()
    return result

# Streamlit UI
st.title("🦷 Dental Image Classification & Disease Info")
st.write("Upload a dental image to classify and get related information.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    with st.spinner("Classifying..."):
        # Make prediction
        predicted_class, predictions = classify_image(image, model)
        
        # Display predictions
        st.subheader("Detection Results:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Detailed Predictions:")
            for class_name, pred in zip(class_labels, predictions):
                st.write(f"{class_name}: {pred*100:.2f}%")
        
        with col2:
            st.write(f"Primary Diagnosis: **{predicted_class}**")
            confidence = predictions[np.argmax(predictions)] * 100
            st.write(f"Confidence: **{confidence:.2f}%**")

    # Show class probabilities as a bar chart
    st.subheader("Class Probabilities:")
    st.bar_chart({label: prob * 100 for label, prob in zip(class_labels, predictions)})

    # Fetch more disease information using CrewAI
    if st.button("Get Disease Information"):
        llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            temperature=0.7,
            api_key=GEMINI_API_KEY
        )
        with st.spinner("Fetching details..."):
            result = DiseasePredictor(llm, predicted_class)
        st.subheader("Medical Report:")
        st.write(result.raw)