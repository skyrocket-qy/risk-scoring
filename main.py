import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    df, columns = gen_user_data()
    train(df, columns)