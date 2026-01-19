import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_production")
    DEBUG = os.environ.get("DEBUG", True)
    
    # Model settings
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/sentiment_model.pkl")
    
    # App settings
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 3