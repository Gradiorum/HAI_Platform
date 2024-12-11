from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import json
import os

with open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "db_config.json"), "r") as f:
    db_config = json.load(f)

DATABASE_URL = db_config["database_url"]

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
