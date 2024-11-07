from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from backend.models.chat import ChatMessage, ChatResponse, ChatSession
from backend.services.chat import ChatService
from backend.core.security import get_current_user

router = APIRouter()
chat_service = ChatService()

@router.post("/sessions", response_model=ChatSession)
async def create_session(current_user: dict = Depends(get_current_user)):
    return await chat_service.create_session(str(current_user["_id"]))

@router.get("/sessions", response_model=List[ChatSession])
async def get_sessions(current_user: dict = Depends(get_current_user)):
    return await chat_service.get_user_sessions(str(current_user["_id"]))

@router.post("/{session_id}/send", response_model=ChatResponse)
async def send_message(
    session_id: str,
    message: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    return await chat_service.process_message(message.text, session_id, current_user)

@router.get("/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    return await chat_service.get_session_messages(session_id, str(current_user["_id"]))