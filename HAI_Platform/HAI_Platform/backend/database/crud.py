from sqlalchemy.orm import Session
from .models import UserLog, Feedback, EyeTrackingData
from .schemas import UserLogCreate, FeedbackCreate, EyeTrackingCreate

def create_user_log(db: Session, log: UserLogCreate):
    db_log = UserLog(
        user_id=log.user_id,
        prompt=log.prompt,
        response=log.response
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def create_feedback(db: Session, fb: FeedbackCreate):
    db_fb = Feedback(
        user_id=fb.user_id,
        interaction_id=fb.interaction_id,
        feedback_type=fb.feedback_type,
        details=fb.details
    )
    db.add(db_fb)
    db.commit()
    db.refresh(db_fb)
    return db_fb

def create_eye_tracking(db: Session, et: EyeTrackingCreate):
    db_et = EyeTrackingData(
        interaction_id=et.interaction_id,
        x=et.x,
        y=et.y
    )
    db.add(db_et)
    db.commit()
    db.refresh(db_et)
    return db_et
