# SmartChat Backend

This is the backend service for the SmartChat application built with FastAPI.

## Project Structure

```
backend/
├── api/
│   └── routes/
│       ├── auth.py
│       └── chat.py
├── core/
│   ├── config.py
│   └── security.py
├── database/
│   └── session.py
├── models/
│   ├── chat.py
│   └── user.py
├── services/
│   ├── auth.py
│   └── chat.py
├── main.py
└── requirements.txt
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc