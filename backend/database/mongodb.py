from motor.motor_asyncio import AsyncIOMotorClient
from backend.core.config import settings
import logging
import os
from typing import Optional
import urllib.parse
import dns.resolver
import ssl

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_db(cls) -> None:
        """Connect to MongoDB/Cosmos DB"""
        try:
            if cls.client is None:
                logging.info("Connecting to MongoDB/Cosmos DB...")
                
                # Get connection string
                connection_string = os.getenv('MONGODB_URL', settings.MONGODB_URL)
                
                if not connection_string:
                    raise ValueError("MongoDB connection string not provided")

                # Connection options for Cosmos DB
                connection_kwargs = {
                    "tls": True,
                    "tlsAllowInvalidCertificates": True,
                    "retryWrites": False,
                    "maxIdleTimeMS": 120000,
                    "serverSelectionTimeoutMS": 30000,
                    "connectTimeoutMS": 30000,
                    "socketTimeoutMS": 30000,
                    "authMechanism": "SCRAM-SHA-256",
                    "directConnection": False,
                    "maxPoolSize": 10,
                    "minPoolSize": 0
                }

                # Create client
                cls.client = AsyncIOMotorClient(
                    connection_string,
                    **connection_kwargs
                )

                # Get database name from environment or default
                db_name = os.getenv('MONGODB_DB', 'smartchat')
                cls.db = cls.client[db_name]

                # Test connection
                await cls.client.admin.command('ping')
                logging.info("Successfully connected to MongoDB/Cosmos DB")

        except Exception as e:
            logging.error(f"MongoDB connection error: {str(e)}")
            if cls.client:
                await cls.close_db()
            raise

    @classmethod
    async def close_db(cls) -> None:
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logging.info("MongoDB connection closed")

    @classmethod
    async def get_db(cls):
        """Get database instance"""
        if cls.db is None:
            await cls.connect_db()
        return cls.db

    @classmethod
    async def get_collection(cls, collection_name: str):
        """Get collection from database"""
        if cls.db is None:
            await cls.connect_db()
        return cls.db[collection_name]

    @classmethod
    def is_connected(cls) -> bool:
        """Check if connected to database"""
        return cls.client is not None and cls.db is not None

    @classmethod
    async def ping(cls) -> bool:
        """Test database connection"""
        try:
            if cls.client:
                await cls.client.admin.command('ping')
                return True
            return False
        except Exception:
            return False