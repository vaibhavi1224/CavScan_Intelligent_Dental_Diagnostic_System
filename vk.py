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

load_dotenv()
GEMINI_API_KEY = "AIzaSyAoXkCAtHMMxphFRSHP1Wc78asPiY-pYC4"

# Load the trained model
MODEL_PATH = "C:\\Users\\Vaibhavi\\Desktop\\cavscann-main\\Model_Assets\\dental_v1_vgg16.h5"

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# Define class labels
class_labels = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth ulcer', 'Tooth Discoloration']

# Function to classify uploaded image
def classify_image(image):
    img = image.resize((256, 256))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, predictions[0]


load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

# CrewAI-powered Disease Information Fetcher
def DiseasePredictor(llm, disease):
    agent = DiseasePredictorAgent()
    task = DiseasePredictorTask()

    researcher_agent = agent.create_disease_researcher(llm, disease, class_labels)
    researcher_task = task.create_disease_researcher_task(researcher_agent, disease)

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
        predicted_label, probabilities = classify_image(image)

    st.subheader("Prediction:")
    st.write(f"🦷 **{predicted_label}**")

    # Show class probabilities as a bar chart
    st.bar_chart({label: prob * 100 for label, prob in zip(class_labels, probabilities)})

    # Fetch more disease information using CrewAI
    if st.button("Get Information"):
        llm = LLM(
            model="gemini/gemini-1.5-pro-latest",
            temperature=0.7,
            api_key=GEMINI_API_KEY
        )
        with st.spinner("Fetching details..."):
            result = DiseasePredictor(llm, predicted_label)
        st.subheader("Disease Information:")
        st.write(result.raw)











from crewai import Agent
from typing import Optional, List

class DiseasePredictorAgent:
    def create_disease_researcher(self, llm, predictions: List, class_labels: List[str], specialization: Optional[str] = None):
        """
        Create a specialized Disease Researcher agent with enhanced capabilities to analyze multiple probabilities.

        Args:
            llm: Language model instance
            predictions: List of confidence scores for each class
            class_labels: Corresponding labels for the predictions
            specialization: Optional specific area of medical specialization
        """
        expertise = specialization if specialization else "Oral and Dental Diseases"

        # ✅ Ensure all predictions are converted to floats, filtering out invalid values
        valid_predictions = []
        valid_labels = []
        
        for i in range(len(class_labels)):
            try:
                confidence = float(predictions[i])  # Ensure numeric conversion
                valid_predictions.append(confidence)
                valid_labels.append(class_labels[i])
            except ValueError:
                print(f"Warning: Skipping invalid prediction '{predictions[i]}' for {class_labels[i]}")

        # ✅ Generate probability strings safely
        disease_probabilities = "\n".join(
            [f"- {valid_labels[i]}: {valid_predictions[i] * 100:.2f}%" for i in range(len(valid_labels))]
        ) if valid_predictions else "No valid predictions available."

        return Agent(
            role=f'Dental AI Specialist - {expertise}',
            goal=f'''As an AI-powered dental specialist, analyze the following diagnostic probabilities:
                {disease_probabilities}

                Generate a **brief and professional dental report** including:
                - Identified dental conditions (if any)
                - A concise assessment of findings
                - Recommended next steps for treatment or further examination.

                Keep the report **clear, structured, and professional, like a dentist’s diagnosis.** Avoid unnecessary details and speculation.
                ''',
            verbose=True,
            backstory=f'''Dr. Sarah Chen is an AI-powered dental expert specializing in early detection of oral diseases.
                - Dental Consultant with expertise in AI-driven diagnostics
                - DMD from Johns Hopkins School of Dentistry
                - Extensive experience in analyzing digital dental scans
                - Collaborates with leading oral health institutions
                ''',
            llm=llm,
            tools=[]
        )




from crewai import Task
from typing import List

class DiseasePredictorTask:
    def create_disease_research_task(self, agent, predictions: List[float], class_labels: List[str]):
        """
        Create a disease research task that analyzes multiple confidence scores from AI predictions.

        Args:
            agent: The AI medical researcher agent
            predictions: List of confidence scores for detected conditions (as floats)
            class_labels: Corresponding labels for each prediction
        """
        
        # ✅ Convert all predictions to floats safely
        valid_predictions = []
        valid_labels = []

        for i in range(len(class_labels)):
          try:
            confidence = float(predictions[i])  # Ensure conversion to float
            valid_predictions.append(confidence)
            valid_labels.append(class_labels[i])
          except ValueError:
           print(f"Warning: Skipping invalid prediction '{predictions[i]}' for {class_labels[i]}")

        # ✅ Ensure valid predictions before applying conditions
        filtered_conditions = [
        f"- {valid_labels[i]}: {valid_predictions[i] * 100:.2f}%"
        for i in range(len(valid_predictions)) if valid_predictions[i] > 0.10  # Only include likely conditions
        ]

        conditions_report = "\n".join(filtered_conditions) 

        # Research Task
        return Task(
            name="AI-Driven Dental Condition Analysis",
            description=f'''
                Analyze AI-generated predictions for dental anomalies and provide a structured diagnostic report.
                
                **Detected conditions with confidence scores:**
                {conditions_report}
                
                **Required analysis components:**
                1. **Condition Overview**:
                   - Definitions and classifications
                   - Epidemiological trends
                   - Common patient demographics
                
                2. **Clinical Insights**:
                   - Pathophysiology and progression
                   - Associated symptoms and risk factors
                   - Potential co-occurrences
                
                3. **Diagnostic Evaluation**:
                   - Recommended tests (imaging, laboratory)
                   - Differential diagnoses
                   - AI vs. traditional diagnosis accuracy
                
                4. **Treatment & Management**:
                   - Available treatment protocols
                   - Evidence-based recommendations
                   - Preventive strategies
                
                5. **Risk Assessment & Prognosis**:
                   - Long-term impact
                   - Risk of complications
                   - Preventive measures
            
            **Inputs:**
            AI Model Predictions (Confidence Scores): {predictions}
            ''',
            expected_output=f'''
                **Deliverables:**
                - Structured diagnostic report covering all detected conditions
                - Confidence analysis and likelihood of co-occurrence
                - Recommended treatment pathways and further testing
                - Insights based on the latest medical research
                
                **Format:**
                - Well-structured medical report
                - Supported by peer-reviewed references
                - Actionable insights for clinical use
            ''',
            agent=agent,
            verbose=True
        )