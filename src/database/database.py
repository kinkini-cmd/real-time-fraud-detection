"""PostgreSQL database operations for storing transaction results."""

from typing import Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


Base = declarative_base()


class TransactionResult(Base):
    """Transaction result model."""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    prediction = Column(String(20), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Integer, nullable=False)
    location = Column(String(100))
    device_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class DatabaseManager:
    """Manages PostgreSQL database operations."""
    
    def __init__(self, host: str = "localhost", port: int = 5432,
                 database: str = "fraud_detection", 
                 user: str = "fraud_user", password: str = "fraud_pass"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.engine = None
        self.SessionLocal = None
    
    def connect(self):
        """Establish database connection."""
        connection_string = (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        print("Database connected and tables created")
    
    def insert_transaction(self, transaction_data: Dict[str, Any]) -> int:
        """Insert a transaction result into the database."""
        if not self.SessionLocal:
            raise RuntimeError("Database not connected")
        
        session = self.SessionLocal()
        try:
            transaction = TransactionResult(
                transaction_id=transaction_data['transaction_id'],
                amount=transaction_data['amount'],
                prediction=transaction_data['prediction'],
                fraud_probability=transaction_data['fraud_probability'],
                is_fraud=transaction_data['is_fraud'],
                location=transaction_data.get('location'),
                device_id=transaction_data.get('device_id')
            )
            session.add(transaction)
            session.commit()
            return transaction.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_recent_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent transactions from the database."""
        if not self.SessionLocal:
            raise RuntimeError("Database not connected")
        
        session = self.SessionLocal()
        try:
            transactions = session.query(TransactionResult)\
                .order_by(TransactionResult.created_at.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    'id': t.id,
                    'transaction_id': t.transaction_id,
                    'amount': t.amount,
                    'prediction': t.prediction,
                    'fraud_probability': t.fraud_probability,
                    'is_fraud': t.is_fraud,
                    'location': t.location,
                    'created_at': t.created_at.isoformat() if t.created_at else None
                }
                for t in transactions
            ]
        finally:
            session.close()
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """Get fraud statistics from the database."""
        if not self.SessionLocal:
            raise RuntimeError("Database not connected")
        
        session = self.SessionLocal()
        try:
            total = session.query(TransactionResult).count()
            fraud = session.query(TransactionResult).filter(
                TransactionResult.is_fraud == 1
            ).count()
            
            return {
                'total_transactions': total,
                'fraud_transactions': fraud,
                'legitimate_transactions': total - fraud,
                'fraud_rate': (fraud / total * 100) if total > 0 else 0
            }
        finally:
            session.close()


if __name__ == "__main__":
    db = DatabaseManager()
    db.connect()
    print("Database setup complete")