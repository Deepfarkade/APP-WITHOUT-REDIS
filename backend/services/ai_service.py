from interpreter import interpreter
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi import HTTPException, status
import os
import certifi
import ssl
import httpx
import openai
from pathlib import Path

class AIService:
    _instance = None
    _executor = ThreadPoolExecutor(max_workers=10)

    def __init__(self):
        # Configure environment first
        self._configure_environment()
        
        # Initialize interpreter settings
        interpreter.auto_run = True
        interpreter.llm.model = "gpt-3.5-turbo"
        interpreter.llm.temperature = 0.7
        interpreter.llm.supports_functions = True
        # Configure interpreter to auto-approve code execution
        interpreter.system_message += """
        Run shell commands with -y so the user doesn't have to confirm them.
        """
        
        # Set custom instructions
        interpreter.custom_instructions = """
        You are 'Maddy', an AI assistant created by EY India GEN AI Engineers. Your primary focus is on:
        1. Supply chain analysis and optimization
        2. Root cause analysis (RCA)
        3. Predictive quality analysis (PQA)
        4. Data summarization and forecasting
        5. Machine learning insights

        Always maintain a professional tone while being helpful and precise in your responses.
        Focus on providing actionable insights and clear explanations.
        """

    def _configure_environment(self):
        """Configure environment settings for OpenAI and SSL"""
        try:
            # Get API key from environment
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")

            # Get system CA certificates path
            ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE', '/etc/ssl/certs/ca-certificates.crt')
            
            # Configure SSL context with system certificates
            ssl_context = ssl.create_default_context(cafile=ca_bundle)
            ssl_context.load_default_certs()
            
            # Configure custom httpx client with proper SSL settings
            client = httpx.Client(
                verify=ca_bundle,
                timeout=30.0,
                http2=True
            )
            
            # Set up OpenAI client with custom httpx client
            openai_client = openai.OpenAI(
                api_key=api_key,
                http_client=client
            )
            
            # Configure interpreter
            interpreter.llm.api_key = api_key
            interpreter.llm.client = openai_client
            
            logging.info("OpenAI configuration completed successfully")
            
        except Exception as e:
            logging.error(f"Failed to configure environment: {str(e)}")
            raise

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = AIService()
        return cls._instance

    async def get_ai_response(self, message: str, user_id: str) -> str:
        """Get AI response asynchronously"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._get_interpreter_response,
                message
            )
            return response
        except Exception as e:
            logging.error(f"AI Service error for user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate AI response"
            )

    def _get_interpreter_response(self, message: str) -> str:
        """Get response from interpreter"""
        try:
            # Use chat method for single response
            response = interpreter.chat(message)
            
            # Extract the last assistant message
            if isinstance(response, list):
                for msg in reversed(response):
                    if msg.get('role') == 'assistant':
                        return msg.get('content', '')
            return str(response)
        except Exception as e:
            logging.error(f"Interpreter error: {str(e)}")
            logging.error("Detailed error information:", exc_info=True)
            raise