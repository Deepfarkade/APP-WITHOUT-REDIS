from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ChatMessage(BaseModel):
    id: Optional[str] = None
    text: str
    sender: str = "user"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    text: str
    sender: str = "bot"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str

class ChatSession(BaseModel):
    id: str
    title: str
    last_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    messages: List[ChatMessage] = []