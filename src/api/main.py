"""FastAPI application for fraud detection."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
from datetime import datetime

from src.models.predict import ModelPredictor
from src.monitoring.monitor import SystemMonitor


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="API for detecting fraudulent financial transactions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionRequest(BaseModel):
    """Request schema for transaction prediction."""
    transaction_id: str
    amount: float = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)
    is_new_device: int = Field(..., ge=0, le=1)
    is_foreign_location: int = Field(..., ge=0, le=1)
    transactions_last_hour: int = Field(..., ge=0)
    previous_transaction_amount: float = Field(..., gt=0)
    location: Optional[str] = None
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)
    is_night: Optional[int] = Field(None, ge=0, le=1)
    amount_change: Optional[float] = None
    amount_ratio: Optional[float] = None
    is_high_amount: Optional[int] = Field(None, ge=0, le=1)
    is_high_frequency: Optional[int] = Field(None, ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    transaction_id: str
    prediction: str
    fraud_probability: float
    is_fraud: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


predictor = ModelPredictor()
monitor = SystemMonitor()


def _require_model():
    """Raise a 503 if the model has not been trained/loaded."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train it with src.models.train."
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: TransactionRequest):
    """Predict if a transaction is fraudulent."""
    _require_model()

    try:
        start = time.perf_counter()
        features = transaction.model_dump(exclude_none=True)
        result = predictor.predict_single(features)
        elapsed = time.perf_counter() - start

        monitor.record_prediction(
            {**result, 'transaction_id': transaction.transaction_id},
            elapsed
        )

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            prediction=result['prediction'],
            fraud_probability=result['fraud_probability'],
            is_fraud=result['is_fraud'],
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        monitor.record_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """Predict for multiple transactions."""
    _require_model()

    try:
        start = time.perf_counter()
        features_list = [t.model_dump(exclude_none=True) for t in transactions]
        results = predictor.predict_batch(features_list)
        elapsed = time.perf_counter() - start

        per_item_time = elapsed / len(results) if results else 0
        for t, r in zip(transactions, results):
            monitor.record_prediction(
                {**r, 'transaction_id': t.transaction_id},
                per_item_time
            )

        return {
            "predictions": [
                {
                    "transaction_id": t.transaction_id,
                    "prediction": r['prediction'],
                    "fraud_probability": r['fraud_probability'],
                    "is_fraud": r['is_fraud']
                }
                for t, r in zip(transactions, results)
            ],
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        monitor.record_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get API and model metrics."""
    return {
        "status": "running",
        "model_type": predictor.model_type,
        "model_loaded": predictor.is_loaded,
        "feature_count": len(predictor.feature_columns) if predictor.feature_columns else 0,
        **monitor.get_all_metrics()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
