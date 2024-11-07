from fastapi import APIRouter, Depends, HTTPException, status, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from backend.models.user import UserCreate, UserResponse
from backend.services.auth import AuthService
from backend.core.security import get_current_user
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from backend.core.config import settings
import logging
from typing import Dict, Any

router = APIRouter()
auth_service = AuthService()

async def send_email_background(
    recipient_email: str,
    subject: str,
    body: str
) -> None:
    try:
        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_USERNAME
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
            
        logging.info(f"Password reset email sent successfully to {recipient_email}")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email"
        )

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, Any]:
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = await auth_service.create_access_token(
        data={"sub": user["email"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "full_name": user["full_name"]
        }
    }

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("Authorization")
    return {"message": "Successfully logged out"}

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    return await auth_service.create_user(user)

@router.post("/forgot-password")
async def forgot_password(
    email: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    try:
        # Generate reset token
        reset_token = await auth_service.send_password_reset_email(email)
        
        reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        
        email_body = f"""
        Hello,
        
        You have requested to reset your password. Please click the link below to reset your password:
        
        {reset_link}
        
        If you did not request this, please ignore this email.
        
        Best regards,
        Your App Team
        """
        
        # Add email sending to background tasks
        background_tasks.add_task(
            send_email_background,
            recipient_email=email,
            subject="Password Reset Request",
            body=email_body
        )
        
        return {
            "success": True,
            "message": "Password reset instructions sent to your email"
        }
    except Exception as e:
        logging.error(f"Password reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )

@router.post("/reset-password")
async def reset_password(token: str, new_password: str) -> Dict[str, str]:
    await auth_service.reset_password(token, new_password)
    return {"message": "Password reset successful"}