# Traditional Risk-Scoring with Unsupervised Model

This is a general example using **unsupervised anomaly detection** for risk scoring with **fake data**.

---

## 1. Prepare the environment

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 2. Generate fake data

```python
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate 1000 normal events (3 features)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 3))

# Generate 20 anomalous events (far from normal)
anomalies = np.random.normal(loc=5, scale=1, size=(20, 3))

# Combine data
X = np.vstack([normal_data, anomalies])

# Create DataFrame
columns = ['feature1', 'feature2', 'feature3']
df = pd.DataFrame(X, columns=columns)
df['label'] = [0]*1000 + [1]*20  # 0=normal, 1=anomaly
```

---

## 3. Train an unsupervised model (Isolation Forest)

```python
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(df[columns])

# Compute anomaly scores
scores = model.decision_function(df[columns])  # higher = more normal
anomaly_scores = -scores  # flip: higher = more anomalous

df['anomaly_score'] = anomaly_scores
```

---

## 4. Set a threshold and label anomalies

```python
threshold = np.percentile(anomaly_scores, 98)  # top 2% most anomalous
df['predicted_label'] = (anomaly_scores > threshold).astype(int)
```

---

## 5. Evaluate results

```python
from sklearn.metrics import classification_report

print(classification_report(df['label'], df['predicted_label']))
```

---

## 6. Visualize

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='feature1', y='feature2', hue='predicted_label', data=df, palette=['green', 'red'])
plt.title('Anomaly Detection with Isolation Forest')
plt.show()
```

---

### Notes:

* `IsolationForest` is unsupervised: it does not use `label` for training.
* `anomaly_score` represents the risk score: higher means more likely to be abnormal.
* Threshold can be tuned depending on desired sensitivity.
* You can replace `IsolationForest` with `OneClassSVM`, `LocalOutlierFactor`, or `Autoencoder` depending on use case.
