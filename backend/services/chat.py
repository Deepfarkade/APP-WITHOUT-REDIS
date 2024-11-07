from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from backend.models.chat import ChatMessage, ChatResponse, ChatSession
from backend.database.mongodb import MongoDB
from datetime import datetime
from bson import ObjectId
import logging
import json
from .ai_service import AIService



class ChatService:
    def __init__(self):
        self.sessions_collection = "chat_sessions"
        self.messages_collection = "chat_messages"
        self.ai_service = AIService.get_instance()

    def _serialize_datetime(self, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO format strings"""
        if isinstance(obj, dict):
            return {key: self._serialize_datetime(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def _deserialize_datetime(self, obj: Any) -> Any:
        """Recursively convert ISO format strings back to datetime objects"""
        if isinstance(obj, dict):
            return {key: self._deserialize_datetime(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_datetime(item) for item in obj]
        elif isinstance(obj, str) and 'T' in obj:
            try:
                return datetime.fromisoformat(obj)
            except ValueError:
                return obj
        return obj

    async def create_session(self, user_id: str) -> ChatSession:
        try:
            sessions = await MongoDB.get_collection(self.sessions_collection)
            
            session = ChatSession(
                id=str(ObjectId()),
                title="New Analysis",
                user_id=str(user_id),
                timestamp=datetime.utcnow(),
                messages=[
                    ChatMessage(
                        id=str(ObjectId()),
                        text="Hello! How can I help you with supply chain analysis today?",
                        sender="bot",
                        timestamp=datetime.utcnow(),
                        session_id=str(ObjectId())
                    )
                ]
            )
            
            session_dict = self._serialize_datetime(session.model_dump())
            await sessions.insert_one(session_dict)
            return session

        except Exception as e:
            logging.error(f"Failed to create chat session: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )

    async def process_message(self, text: str, session_id: str, user: dict) -> ChatResponse:
        try:
            sessions = await MongoDB.get_collection(self.sessions_collection)
            
            # Verify session exists and belongs to user
            session = await sessions.find_one({
                "id": session_id,
                "user_id": str(user["_id"])
            })
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            # Save user message
            user_message = ChatMessage(
                id=str(ObjectId()),
                text=text,
                sender="user",
                session_id=session_id,
                timestamp=datetime.utcnow()
            )
            
            # Get AI response
            ai_response_text = await self.ai_service.get_ai_response(text, str(user["_id"]))
            
            # Create response object
            response = ChatResponse(
                id=str(ObjectId()),
                text=ai_response_text,
                sender="bot",
                session_id=session_id,  # Changed from sessionId to session_id
                timestamp=datetime.utcnow()
            )
            
            # Serialize messages for MongoDB storage
            user_message_dict = self._serialize_datetime(user_message.model_dump())
            response_dict = self._serialize_datetime(response.model_dump())
            
            # Update session with messages
            await sessions.update_one(
                {"id": session_id},
                {
                    "$push": {"messages": {
                        "$each": [user_message_dict, response_dict]
                    }},
                    "$set": {
                        "last_message": response.text,
                        "timestamp": datetime.utcnow()
                    }
                }
            )
            
            return response

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to process message: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process message"
            )

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        try:
            sessions = await MongoDB.get_collection(self.sessions_collection)
            cursor = sessions.find({"user_id": user_id}).sort("timestamp", -1)
            
            user_sessions = []
            async for session in cursor:
                # Deserialize datetime objects from the stored session
                deserialized_session = self._deserialize_datetime(session)
                user_sessions.append(ChatSession(**deserialized_session))
            
            return user_sessions

        except Exception as e:
            logging.error(f"Failed to get user sessions: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch chat sessions"
            )

    async def get_session_messages(self, session_id: str, user_id: str) -> List[ChatMessage]:
        try:
            sessions = await MongoDB.get_collection(self.sessions_collection)
            session = await sessions.find_one({
                "id": session_id,
                "user_id": user_id
            })
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            # Deserialize datetime objects from the stored messages
            deserialized_session = self._deserialize_datetime(session)
            return [ChatMessage(**msg) for msg in deserialized_session.get("messages", [])]

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to get session messages: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch session messages"
            )

    async def delete_session(self, session_id: str, user_id: str) -> None:
        try:
            sessions = await MongoDB.get_collection(self.sessions_collection)
            result = await sessions.delete_one({
                "id": session_id,
                "user_id": user_id
            })
            
            if result.deleted_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to delete session: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete chat session"
            )
