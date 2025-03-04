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