
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

scaler = joblib.load(BASE_DIR / "scaler.pkl")
kmeans = joblib.load(BASE_DIR / "kmeans_model.pkl")

app = FastAPI(
    title="Customer Segmentation API",
    description="Predict customer segments using K-Means",
    version="1.0.0"
)


class Customer(BaseModel):
    annual_income: float = Field(gt=0)
    spending_score: float = Field(ge=1, le=100)


@app.get("/")
def home():
    return {
        "message": "Customer Segmentation API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_cluster(customer: Customer):
    data = np.array([[
        customer.annual_income,
        customer.spending_score
    ]])

    scaled_data = scaler.transform(data)
    cluster = int(kmeans.predict(scaled_data)[0])

    return {
        "annual_income": customer.annual_income,
        "spending_score": customer.spending_score,
        "cluster": cluster,
        "message": f"Customer belongs to Cluster {cluster}"
    }
