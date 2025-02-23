from fastapi import FastAPI
from app.api import websocket
from app.middleware.websocket import setup_websocket_middleware

app = FastAPI(title="ML Service API")

# Setup middleware
setup_websocket_middleware(app)

# Include WebSocket routes
app.include_router(websocket.router) 