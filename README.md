# Real-Time Fraud Detection System

An end-to-end Machine Learning Engineering project that detects potentially fraudulent financial transactions in near real time.

## Features

- 🤖 Machine Learning model for fraud detection (Random Forest, XGBoost)
- 📊 Real-time transaction streaming with Apache Kafka
- 🧪 Feature engineering for transaction patterns
- 🚀 FastAPI REST API for predictions
- 💾 PostgreSQL for transaction storage
- 📈 MLflow for experiment tracking and model versioning
- 🐳 Docker and Docker Compose for containerization
- 📊 Streamlit dashboard for monitoring
- 🔍 Data drift and model drift detection
- ✅ Automated testing with Pytest
- 🔄 CI/CD pipeline with GitHub Actions
- ☁️ Cloud deployment ready

## Architecture

```
Transaction → Kafka → Data Validation → Feature Engineering → ML Model → Fraud Probability
                                                                          ↓
                        Legitimate           Fraud
                            ↓                 ↓
                         Approve            Block/Alert
                            ↓                 ↓
                         PostgreSQL
                            ↓
                       Monitoring Dashboard
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd real-time-fraud-detection
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python -m src.models.train
```

This saves the trained model to `models/fraud_model.pkl` and logs the run to MLflow.

5. Start the API:
```bash
uvicorn src.api.main:app --reload
```

6. Start the dashboard (optionally point it at a non-default API):
```bash
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

7. (Optional) Stream transactions through Kafka end to end:
```bash
# Terminal 1: consume, score, persist, and monitor transactions
python -m src.kafka.pipeline

# Terminal 2: publish sample transactions
python -m src.kafka.producer
```

### Using Docker Compose

Start the full stack:
```bash
docker-compose up -d
```

Services:
- FastAPI: http://localhost:8000
- Streamlit Dashboard: http://localhost:8501
- MLflow: http://localhost:5000
- PostgreSQL: localhost:5432
- Kafka: localhost:9092

## Project Structure

```
real-time-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   ├── kafka/
│   ├── database/
│   └── monitoring/
├── models/
├── tests/
├── dashboard/
├── docker/
├── .github/workflows/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Documentation

Once the API is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

- `GET /health` - Health check (reports whether the model is loaded)
- `POST /predict` - Predict fraud for a single transaction
- `POST /predict/batch` - Predict fraud for multiple transactions
- `GET /metrics` - Live ML/system metrics plus model information

The API returns `503` from prediction endpoints until `models/fraud_model.pkl`
exists. Train the model first (step 4 above).

## Testing

```bash
pytest tests/ -v
```

## CI/CD

The `.github/workflows/ci.yml` file defines a CI pipeline that:
1. Installs dependencies
2. Runs linting
3. Executes tests
4. Trains the model
5. Builds Docker image

## Monitoring

The system monitors:
- API response time
- Request count
- Error rate
- Fraud rate
- Model performance (precision, recall, F1)
- Data drift
- System resources (CPU, memory)

## Technologies Used

- Python, Pandas, NumPy, Scikit-learn, XGBoost
- FastAPI, Uvicorn, Pydantic
- Apache Kafka
- PostgreSQL, SQLAlchemy
- MLflow
- Docker, Docker Compose
- Streamlit
- Pytest
- GitHub Actions

## Learning Outcomes

This project demonstrates:
- Machine Learning (classification, imbalanced datasets, feature engineering)
- Data Engineering (streaming, Kafka, event-driven architecture)
- MLOps (MLflow, model versioning, monitoring, CI/CD)
- Backend Development (REST APIs, FastAPI)
- DevOps (Docker, containerization, cloud deployment)

## License

MIT