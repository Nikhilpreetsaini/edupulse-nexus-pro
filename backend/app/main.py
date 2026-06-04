"""
EduPulse Nexus Pro Backend with Demo Mode and Health Endpoints

This FastAPI application extends the original EduPulse Nexus API with a few
additional endpoints to support professor demo scenarios and basic health
monitoring.  It continues to expose the core upload/prediction/statistics
APIs while adding:

    * GET /         – Root endpoint to verify the API is running.
    * GET /health   – Return high‑level API, model and dataset status.
    * POST /seed_demo – Load a synthetic professor demo dataset and train
      the model on it.  This allows the front‑end to demonstrate the full
      workflow without requiring a manual upload.

The demo dataset is generated on the fly with 120 records evenly split
across Low, Medium and High risk labels.  Numeric features are sampled
from representative ranges based on the problem description.  See
generate_demo_dataset() for details.

This file is meant to live under ``backend/app/main.py`` in the
edupulse‑nexus‑pro repository.  It is deliberately self contained so that
adding it via GitHub's contents API will produce a working backend.
"""

from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import io
import random


app = FastAPI(
    title="EduPulse Nexus Pro API",
    description="Backend for the EduPulse Nexus Pro platform with demo mode and health checks",
)

# Allow requests from any origin for development; adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded dataset and trained model.
DATASET: Optional[pd.DataFrame] = None
MODEL: Optional[LogisticRegression] = None
FEATURE_COLUMNS: List[str] = []


class StudentFeatures(BaseModel):
    """Pydantic model representing student input features for prediction."""

    attendancePercentage: float
    studyHoursPerWeek: float
    previousGrade: float
    assignmentCompletionRate: float
    quizAverage: float
    labPerformance: float
    internalAssessmentScore: float
    participationScore: float
    sleepHours: float
    stressLevel: float
    extracurricularLoad: float
    internetAccessQuality: Optional[float] = None  # optional

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the features into a single‑row DataFrame in the order of FEATURE_COLUMNS."""
        data = self.dict()
        # Replace None with NaN which the model can handle
        for key, value in data.items():
            if value is None:
                data[key] = np.nan
        return pd.DataFrame([data])


def train_model(df: pd.DataFrame) -> None:
    """Train a logistic regression model on the provided dataset.

    The target column should be named 'riskLevel' and contain labels such as
    'Low', 'Medium' and 'High'.  All other numeric columns are used as
    features.  The trained model and feature columns are stored in the module
    globals for later predictions.
    """
    global MODEL, FEATURE_COLUMNS, DATASET

    # Drop rows with missing target
    df = df.dropna(subset=["riskLevel"])

    # Identify feature columns (all except target and non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "riskLevel" in numeric_cols:
        numeric_cols.remove("riskLevel")

    FEATURE_COLUMNS = numeric_cols

    if not FEATURE_COLUMNS:
        raise ValueError("No numeric feature columns found in the dataset.")

    # Create X and y
    X = df[FEATURE_COLUMNS]
    y = df["riskLevel"]

    # Split dataset for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode target labels into integers
    y_train_encoded = y_train.map({"Low": 0, "Medium": 1, "High": 2}).values
    y_test_encoded = y_test.map({"Low": 0, "Medium": 1, "High": 2}).values

    # Train logistic regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train_encoded)

    # Evaluate model
    y_pred = model.predict(X_test)
    app.state.model_metrics = {
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "precision": float(precision_score(y_test_encoded, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test_encoded, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0)),
    }

    # Store global state
    MODEL = model
    DATASET = df

    # Save the model to disk for persistence (optional)
    try:
        joblib.dump({"model": MODEL, "feature_columns": FEATURE_COLUMNS}, "/home/oai/share/edupulse_model.joblib")
    except Exception:
        # If running in an environment without filesystem access, ignore persistence
        pass


def categorize_prediction(prob: float) -> str:
    """Convert a probability to risk category.

    Uses simple thresholds: <0.33 → Low, <0.66 → Medium, otherwise High.
    """
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"



def generate_student_record(risk: str) -> dict:
    """Generate a single student record with numeric features sampled
    from ranges representative of the given risk category.
    """
    if risk == "Low":
        return {
            "attendancePercentage": random.uniform(80, 100),
            "studyHoursPerWeek": random.uniform(14, 20),
            "previousGrade": random.uniform(80, 100),
            "assignmentCompletionRate": random.uniform(80, 100),
            "quizAverage": random.uniform(80, 100),
            "labPerformance": random.uniform(80, 100),
            "internalAssessmentScore": random.uniform(80, 100),
            "participationScore": random.uniform(80, 100),
            "sleepHours": random.uniform(7, 9),
            "stressLevel": random.uniform(1, 4),
            "extracurricularLoad": random.uniform(1, 5),
            "internetAccessQuality": random.uniform(7, 10),
            "riskLevel": "Low",
        }
    if risk == "Medium":
        return {
            "attendancePercentage": random.uniform(60, 80),
            "studyHoursPerWeek": random.uniform(8, 13),
            "previousGrade": random.uniform(65, 85),
            "assignmentCompletionRate": random.uniform(60, 80),
            "quizAverage": random.uniform(60, 80),
            "labPerformance": random.uniform(60, 80),
            "internalAssessmentScore": random.uniform(60, 80),
            "participationScore": random.uniform(60, 80),
            "sleepHours": random.uniform(6, 8),
            "stressLevel": random.uniform(4, 7),
            "extracurricularLoad": random.uniform(2, 6),
            "internetAccessQuality": random.uniform(5, 8),
            "riskLevel": "Medium",
        }
    # High risk
    return {
        "attendancePercentage": random.uniform(40, 60),
        "studyHoursPerWeek": random.uniform(3, 7),
        "previousGrade": random.uniform(50, 70),
        "assignmentCompletionRate": random.uniform(50, 70),
        "quizAverage": random.uniform(50, 70),
        "labPerformance": random.uniform(50, 70),
        "internalAssessmentScore": random.uniform(50, 70),
        "participationScore": random.uniform(50, 70),
        "sleepHours": random.uniform(4, 6),
        "stressLevel": random.uniform(7, 10),
        "extracurricularLoad": random.uniform(3, 7),
        "internetAccessQuality": random.uniform(3, 6),
        "riskLevel": "High",
    }



def generate_demo_dataset() -> pd.DataFrame:
    """Create a synthetic demo dataset of 120 students (40 per risk class).

    The ranges for each feature are chosen to loosely follow the risk factor
    relationships described in the project specification.  This function is
    deterministic only in the number of samples and class distribution; the
    numeric values are randomly sampled on each call.
    """
    records = []
    for _ in range(40):
        records.append(generate_student_record("Low"))
    for _ in range(40):
        records.append(generate_student_record("Medium"))
    for _ in range(40):
        records.append(generate_student_record("High"))
    return pd.DataFrame(records)


@app.get("/")
async def root():
    """Root endpoint returning a simple status message."""
    return {
        "message": "EduPulse Nexus Pro backend is running",
        "api_version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Return health information about the API, model and dataset."""
    return {
        "api": "online",
        "model_trained": MODEL is not None,
        "dataset_loaded": DATASET is not None,
        "records": int(len(DATASET)) if DATASET is not None else 0,
        "feature_columns": FEATURE_COLUMNS,
    }


@app.post("/seed_demo")
async def seed_demo():
    """Load a synthetic demo dataset and train the model on it.

    Returns summary statistics similar to /upload_dataset.  This endpoint
    allows the frontend to instantly populate the dashboard and other pages
    without manual file upload.  Each call replaces the currently loaded
    dataset and model.
    """
    try:
        df = generate_demo_dataset()
        train_model(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    stats = {
        "records": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": FEATURE_COLUMNS,
        "target_distribution": df["riskLevel"].value_counts().to_dict(),
        "model_metrics": app.state.model_metrics,
    }
    return stats


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset and train the model.

    The CSV must contain a column named 'riskLevel' with categorical labels.
    Returns summary statistics and model performance.
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read CSV file")

    if "riskLevel" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must contain a 'riskLevel' column")

    # Train model and store dataset
    try:
        train_model(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compute simple stats for summary (e.g., number of records)
    stats = {
        "records": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": FEATURE_COLUMNS,
        "target_distribution": df["riskLevel"].value_counts().to_dict(),
        "model_metrics": app.state.model_metrics,
    }
    return stats


@app.get("/students")
async def get_students():
    """Return all student records currently in memory.

    The dataset is loaded from the last uploaded file or demo mode.  This
    endpoint returns the records as a list of dicts, excluding the
    'riskLevel' target column.
    """
    if DATASET is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    # Remove target column to avoid leaking original risk
    df_no_target = DATASET.drop(columns=["riskLevel"], errors="ignore")
    return df_no_target.to_dict(orient="records")


@app.post("/predict")
async def predict_risk(features: StudentFeatures):
    """Predict risk for a single student feature set.

    Returns the probability for the high-risk class and the categorical risk level.
    """
    if MODEL is None or not FEATURE_COLUMNS:
        raise HTTPException(status_code=400, detail="Model has not been trained yet. Please upload a dataset or seed demo first.")

    # Convert input features to DataFrame
    df_in = features.to_dataframe()

    # Reorder columns to match training features; fill missing columns with NaN
    for col in FEATURE_COLUMNS:
        if col not in df_in.columns:
            df_in[col] = np.nan
    df_in = df_in[FEATURE_COLUMNS]

    # Simple imputation: replace NaN with column means from dataset
    imputed = df_in.copy()
    for col in FEATURE_COLUMNS:
        if imputed[col].isna().any():
            col_mean = DATASET[col].mean() if DATASET is not None else 0.0
            imputed[col].fillna(col_mean, inplace=True)

    # Predict probabilities for each class and take the probability for 'High' (index 2)
    probs = MODEL.predict_proba(imputed)[0]
    prob_high = float(probs[2])
    risk_level = categorize_prediction(prob_high)
    return {"probability_high": prob_high, "riskLevel": risk_level}


@app.get("/statistics")
async def get_statistics():
    """Return basic aggregate statistics about the dataset.

    Includes the number of records, numeric column means and standard deviations.
    """
    if DATASET is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    summary = {
        "records": int(len(DATASET)),
        "numeric_means": DATASET[FEATURE_COLUMNS].mean().to_dict(),
        "numeric_stddev": DATASET[FEATURE_COLUMNS].std().to_dict(),
        "target_distribution": DATASET["riskLevel"].value_counts().to_dict(),
    }
    return summary


@app.get("/model_insights")
async def model_insights():
    """Return simple model performance metrics recorded during the last training."""
    if not hasattr(app.state, "model_metrics"):
        raise HTTPException(status_code=404, detail="Model metrics not available")
    return app.state.model_metrics
