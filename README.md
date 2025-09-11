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
# user id, role, group, ip, city, region, device gingerprint, time of login, login method, mfa used, 
# login freq, session duration, failed login attempt, login velocity, new device or ip, 
# vpn/proxy use, ASN/ISP, browser or OS anomalies
# Distance from last login location, Time delta from usual login hours, Device churn
# IP risk score, Embedding of login context
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
```

---

## 3. Train an unsupervised models

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
