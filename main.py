import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Define the path for the trained model
MODEL_PATH = "isolation_forest_model.joblib"

def gen_user_data():
    # Set seed for reproducibility
    np.random.seed(42)

    # Number of samples
    n_samples = 1000

    # Generate synthetic features
    # feature1 and feature2 will be used for anomaly detection
    feature1 = np.random.randn(n_samples) * 10 + 50  # e.g., login frequency
    feature2 = np.random.randn(n_samples) * 5 + 30   # e.g., session duration
    feature3 = np.random.randint(0, 2, n_samples)    # e.g., MFA used (0 or 1)
    feature4 = np.random.rand(n_samples)             # e.g., IP risk score (0 to 1)

    # Introduce some anomalies
    # High login frequency and short session duration
    feature1[10:20] = np.random.randn(10) * 5 + 100
    feature2[10:20] = np.random.randn(10) * 2 + 5

    # Low login frequency and long session duration
    feature1[30:40] = np.random.randn(10) * 5 + 10
    feature2[30:40] = np.random.randn(10) * 2 + 60

    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(n_samples),
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'label': 0 # Default label, will be updated for anomalies if needed for evaluation
    })

    # For demonstration, let's assume we know some true anomalies (for evaluation purposes)
    # In a real unsupervised scenario, you wouldn't have this 'label' column for training
    df.loc[10:20, 'label'] = 1
    df.loc[30:40, 'label'] = 1

    columns = ['feature1', 'feature2', 'feature3', 'feature4']
    return df, columns

def train(df, columns):
    # Train Isolation Forest
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(df[columns])

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    # Compute anomaly scores
    scores = model.decision_function(df[columns])  # higher = more normal
    anomaly_scores = -scores  # flip: higher = more anomalous

    df['anomaly_score'] = anomaly_scores

    # Set a threshold and label anomalies
    threshold = np.percentile(anomaly_scores, 98)  # top 2% most anomalous
    df['predicted_label'] = (anomaly_scores > threshold).astype(int)

    # Evaluate results (if 'label' column exists for true anomalies)
    if 'label' in df.columns:
        print("Classification Report:")
        print(classification_report(df['label'], df['predicted_label']))

    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='feature1', y='feature2', hue='predicted_label', data=df, palette=['green', 'red'], alpha=0.7)
    plt.title('Anomaly Detection with Isolation Forest')
    plt.xlabel('Feature 1 (e.g., Login Frequency)')
    plt.ylabel('Feature 2 (e.g., Session Duration)')
    plt.show()

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        return None

# FastAPI Application
app = FastAPI()

# Load the model when the application starts
# In a real-world scenario, you might want to train the model periodically
# or load a pre-trained model from a persistent storage.
ml_model = None
columns_for_prediction = ['feature1', 'feature2', 'feature3', 'feature4']

@app.on_event("startup")
async def startup_event():
    global ml_model
    ml_model = load_model()
    if ml_model is None:
        # If model not found, train it
        print("Training model on startup...")
        df, columns = gen_user_data()
        train(df, columns)
        ml_model = load_model()

class LoginData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
async def predict_anomaly(data: LoginData):
    if ml_model is None:
        return {"error": "Model not loaded. Please ensure the model is trained."}

    # Convert input data to a format suitable for the model
    input_df = pd.DataFrame([data.model_dump()])

    # Ensure the order of columns matches the training data
    input_df = input_df[columns_for_prediction]

    # Predict anomaly score
    scores = ml_model.decision_function(input_df)
    anomaly_score = -scores[0]  # Higher = more anomalous

    # You can also return a predicted label based on a threshold if needed
    # For simplicity, we'll just return the anomaly score
    return {"anomaly_score": anomaly_score}

# To run this FastAPI application, save it as main.py and run:
# uvicorn main:app --reload --port 8000
