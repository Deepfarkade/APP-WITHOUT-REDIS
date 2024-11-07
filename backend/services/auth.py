from datetime import datetime, timedelta
from fastapi import HTTPException, status
from backend.core.config import settings
from backend.database.mongodb import MongoDB
from passlib.context import CryptContext
from jose import jwt
from backend.models.user import UserCreate, UserInDB, UserResponse
import logging
from typing import Optional, Dict, Any

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.users_collection = "users"

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        try:
            users = await MongoDB.get_collection(self.users_collection)
            return await users.find_one({"email": email})
        except Exception as e:
            logging.error(f"Error fetching user by email: {str(e)}")
            return None

    async def create_user(self, user: UserCreate) -> UserResponse:
        try:
            users = await MongoDB.get_collection(self.users_collection)
            
            if await self.get_user_by_email(user.email):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            user_dict = user.model_dump()
            user_dict["hashed_password"] = self.get_password_hash(user_dict.pop("password"))
            user_dict["created_at"] = datetime.utcnow()
            user_dict["is_active"] = True
            
            result = await users.insert_one(user_dict)
            user_dict["id"] = str(result.inserted_id)
            
            return UserResponse(**user_dict)
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        try:
            user = await self.get_user_by_email(email)
            if not user:
                return None
            if not self.verify_password(password, user["hashed_password"]):
                return None
            if not user.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User account is disabled"
                )
            return user
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Authentication error: {str(e)}")
            return None

    async def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access"
            })
            
            return jwt.encode(
                to_encode,
                settings.SECRET_KEY,
                algorithm=settings.ALGORITHM
            )
        except Exception as e:
            logging.error(f"Token creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )

    async def send_password_reset_email(self, email: str) -> str:
        try:
            user = await self.get_user_by_email(email)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Generate reset token
            reset_token = jwt.encode(
                {
                    "sub": email,
                    "exp": datetime.utcnow() + timedelta(hours=1),
                    "type": "reset"
                },
                settings.SECRET_KEY,
                algorithm=settings.ALGORITHM
            )
            
            # Store reset token in MongoDB
            users = await MongoDB.get_collection(self.users_collection)
            await users.update_one(
                {"email": email},
                {"$set": {"reset_token": reset_token, "reset_token_exp": datetime.utcnow() + timedelta(hours=1)}}
            )
            
            return reset_token
        except Exception as e:
            logging.error(f"Password reset error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process password reset request"
            )

    async def reset_password(self, token: str, new_password: str) -> None:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            email = payload.get("sub")
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid token"
                )
            
            users = await MongoDB.get_collection(self.users_collection)
            user = await users.find_one({
                "email": email,
                "reset_token": token,
                "reset_token_exp": {"$gt": datetime.utcnow()}
            })
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or expired token"
                )
            
            # Update password and remove reset token
            hashed_password = self.get_password_hash(new_password)
            await users.update_one(
                {"email": email},
                {
                    "$set": {"hashed_password": hashed_password},
                    "$unset": {"reset_token": "", "reset_token_exp": ""}
                }
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token"
            )
