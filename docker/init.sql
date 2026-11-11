-- Initialize database schema for fraud detection

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL,
    amount NUMERIC(15, 2) NOT NULL,
    prediction VARCHAR(20) NOT NULL,
    fraud_probability FLOAT NOT NULL,
    is_fraud INTEGER NOT NULL DEFAULT 0,
    location VARCHAR(100),
    device_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_transaction_id ON transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);

-- Create fraud alerts table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL,
    amount NUMERIC(15, 2) NOT NULL,
    fraud_probability FLOAT NOT NULL,
    alert_reason TEXT,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    avg_response_time FLOAT,
    total_predictions INTEGER,
    fraud_rate FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO transactions (transaction_id, amount, prediction, fraud_probability, is_fraud, location, created_at)
VALUES 
    ('TX001', 450000.00, 'FRAUD', 0.978, 1, 'Foreign', CURRENT_TIMESTAMP - INTERVAL '1 hour'),
    ('TX002', 2500.00, 'LEGITIMATE', 0.012, 0, 'Columbo', CURRENT_TIMESTAMP - INTERVAL '2 hours'),
    ('TX003', 320000.00, 'FRAUD', 0.895, 1, 'Foreign', CURRENT_TIMESTAMP - INTERVAL '3 hours')
ON CONFLICT DO NOTHING;