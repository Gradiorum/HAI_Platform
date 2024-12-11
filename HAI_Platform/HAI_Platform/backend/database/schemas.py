from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserLogCreate(BaseModel):
    user_id: Optional[str] = "anonymous"
    prompt: str
    response: str

class FeedbackCreate(BaseModel):
    user_id: Optional[str] = "anonymous"
    interaction_id: int
    feedback_type: str
    details: str

class EyeTrackingCreate(BaseModel):
    interaction_id: int
    x: float
    y: float
