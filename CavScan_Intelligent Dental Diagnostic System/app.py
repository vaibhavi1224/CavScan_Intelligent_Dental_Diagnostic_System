import streamlit as st
import os
import torch
import tempfile
from Agent.agents import DiseasePredictorAgent
from Agent.tasks import DiseasePredictorTask
import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from crewai import Crew, Process, LLM
import sqlite3

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# CrewAI-powered Disease Information Fetcher
def DiseasePredictor(llm, disease):
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

# Database setup
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
""")
conn.commit()

def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone() is not None

# Load the trained model
MODEL_PATH = "Model_Assets/dental_v1_vgg16.h5"

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# Define class labels
class_labels = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth ulcer', 'Tooth Discoloration']

def classify_image(image):
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    return class_labels[predicted_class_index], predictions[0]

# Session state for authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Authentication Page
if not st.session_state["logged_in"]:
    st.title("CavScan: AI-Powered Dental Diagnosis")
    st.write("Welcome to CavScan, an AI-powered system for automated dental disease detection and reporting.")
    
    auth_option = st.radio("Choose an option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if auth_option == "Register":
        if st.button("Create Account"):
            if register_user(username, password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists.")
    else:
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.experimental_user()
            else:
                st.error("Invalid username or password")
    st.stop()

# Main Application (After Login)
st.sidebar.title(f"Welcome, {st.session_state['username']}")
st.sidebar.write("Upload patient data and images for diagnosis.")

st.title("Dental Diagnosis System")

# Collect patient details
st.header("Patient Information")
patient_name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120, step=1)
symptoms = st.text_area("Symptoms")

# Image Upload
st.header("Upload Dental X-Ray or Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        predicted_label, probabilities = classify_image(image)
    
    st.subheader("Prediction:")
    st.write(f"🦷 **{predicted_label}**")
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

st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
