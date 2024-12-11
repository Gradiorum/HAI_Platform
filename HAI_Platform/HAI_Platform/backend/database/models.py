from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class UserLog(Base):
    __tablename__ = "user_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="anonymous")
    prompt = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="anonymous")
    interaction_id = Column(Integer, ForeignKey("user_logs.id"))
    feedback_type = Column(String)  # e.g. "accept", "reject", "ranking", "dpo_corrected"
    details = Column(Text)

    interaction = relationship("UserLog", backref="feedback")

class EyeTrackingData(Base):
    __tablename__ = "eye_tracking_data"
    id = Column(Integer, primary_key=True, index=True)
    interaction_id = Column(Integer, ForeignKey("user_logs.id"))
    x = Column(Float)
    y = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    interaction = relationship("UserLog", backref="eye_tracking")
