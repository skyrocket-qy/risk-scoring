# Traditional Risk-Scoring with Unsupervised Model

This project demonstrates a traditional risk-scoring system using **unsupervised anomaly detection** with **fake data**, served via a FastAPI application.

---

## Project Setup and Running

### 1. Prepare the Environment

It's recommended to use a Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]" uvicorn joblib numpy pandas scikit-learn matplotlib seaborn
```

### 2. Train the Model (Initial Setup)

The `main.py` script includes logic to generate synthetic data, train an Isolation Forest model, and save it. If the model file (`isolation_forest_model.joblib`) does not exist when the FastAPI server starts, it will automatically train and save the model.

You can also explicitly run the training and visualization part (without starting the server) by executing:

```bash
.venv/bin/python main.py
```

### 3. Run the FastAPI Server

The risk-scoring model is exposed via a FastAPI application.

```bash
.venv/bin/fastapi run main.py --port 8000 --reload
```

The server will be accessible at `http://localhost:8000`.

### 4. Test the API Endpoint

Once the server is running, you can send a POST request to the `/predict` endpoint to get an anomaly score.

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "feature1": 55.0,
  "feature2": 32.0,
  "feature3": 1.0,
  "feature4": 0.5
}'
```

---

## Model and Approach

This project uses an **Isolation Forest** model for unsupervised anomaly detection.

*   `IsolationForest` is unsupervised: it does not use `label` for training.
*   `anomaly_score` represents the risk score: higher means more likely to be abnormal.
*   Threshold for labeling anomalies can be tuned depending on desired sensitivity.
*   Other unsupervised models like `OneClassSVM`, `LocalOutlierFactor`, or `Autoencoder` could also be used.

---

## Iterative Training Flow (Human-in-the-Loop)

This project incorporates a human-in-the-loop approach for continuous model improvement:

1.  **Initial Training:** The model is trained on all available data (including both normal and potentially abnormal data).
2.  **Anomaly Identification:** The model identifies and flags data points that appear "far" or anomalous.
3.  **Admin Review:** An administrator reviews the flagged data points.
    *   If confirmed as a true anomaly, this information is used to refine the model's understanding of anomalies.
    *   If it's a false positive (a normal event incorrectly flagged), this feedback helps the model better understand normal behavior.
4.  **Model Retraining:** The model is retrained using the updated and refined dataset, incorporating the human feedback. This iterative process helps the model adapt to changing data patterns and improve its accuracy over time.
