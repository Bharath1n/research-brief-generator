from fastapi import FastAPI, Body, HTTPException
from .graph import app
from .schemas import FinalBrief
from .state import AppState
from .history import load_history
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_app = FastAPI()

@api_app.post("/brief", response_model=FinalBrief)
async def generate_brief(body: dict = Body(...)):
    try:
        # Validate required fields
        required_fields = {"topic", "depth", "follow_up", "user_id"}
        missing_fields = required_fields - set(body.keys())
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_fields}")

        # Initialize AppState with default context_summary
        inputs = AppState(
            topic=body["topic"],
            depth=body["depth"],
            follow_up=body["follow_up"],
            user_id=body["user_id"],
            context_summary=""
        )
        config = {"configurable": {"thread_id": body["user_id"]}}
        result = app.invoke(inputs, config=config)
        return result["final_brief"]
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating brief: {str(e)}")

@api_app.get("/brief")
async def get_brief(user_id: str = None):
    if not user_id:
        return {"message": "Use POST /brief to generate a brief. Provide user_id to retrieve history (optional)."}
    try:
        history = load_history(user_id)
        if not history:
            return {"message": f"No briefs found for user_id: {user_id}"}
        return {"user_id": user_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving brief history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving brief: {str(e)}")

@api_app.get("/history/{user_id}")
async def get_history(user_id: str):
    history = load_history(user_id)
    if not history:
        raise HTTPException(404, "No history found")
    return {"history": history}

@api_app.get("/")
async def root():
    return {"message": "Research Brief Generator API"}