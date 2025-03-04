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
