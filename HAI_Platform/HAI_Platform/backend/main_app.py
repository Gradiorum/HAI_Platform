from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from sqlalchemy.orm import Session
from .database.db import get_db, Base, engine
from .database.schemas import UserLogCreate, FeedbackCreate
from .database.crud import create_user_log, create_feedback
from .inference.inference_manager import InferenceManager
from .websocket_manager import manager
import json

Base.metadata.create_all(bind=engine)

app = FastAPI()
inference_manager = InferenceManager()

@app.post("/chat")
def chat(prompt: str, model_name: str = "gpt-4", method: str = "none", user_id: str = "anonymous", db: Session = Depends(get_db)):
    response = inference_manager.run_inference(prompt, model_name, method)
    log = create_user_log(db, UserLogCreate(user_id=user_id, prompt=prompt, response=response))
    return {"prompt": prompt, "response": response, "interaction_id": log.id}

@app.post("/feedback")
def feedback(interaction_id: int, feedback_type: str, details: str, user_id: str = "anonymous", db: Session = Depends(get_db)):
    fb = create_feedback(db, FeedbackCreate(user_id=user_id, interaction_id=interaction_id, feedback_type=feedback_type, details=details))
    return {"status": "success", "feedback_id": fb.id}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get('type') == 'eye_tracking':
                # Placeholder for eye-tracking data handling
                pass
    except WebSocketDisconnect:
        manager.disconnect(client_id)
