"""
Alice in Cyberland - WebSocket Chat Server
Handles real-time chat with Ollama, TTS, and video state management
"""

import asyncio
from contextlib import asynccontextmanager
import json
import uuid
from typing import Dict, Set, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .tts_engine import AliceTTSEngine
import yaml
import requests

# Load configuration
with open('data/video_manifest.yaml', 'r') as f:
    MANIFEST = yaml.safe_load(f)

# Chat state manager
class ChatStateManager:
    def __init__(self):
        self.tts_engine = AliceTTSEngine()
        self.state_transitions = MANIFEST.get('state_transitions', {})
        self.video_states = MANIFEST.get('video_clips', {})
        self.active_clients: Set[WebSocket] = set()
        self.current_video_state = "idle"

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for websocket in self.active_clients:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Broadcast error: {e}")
                disconnected.add(websocket)

        # Remove disconnected clients
        self.active_clients -= disconnected

    def get_video_state_for_message(self, message: str) -> str:
        """Determine appropriate video state based on message content"""
        message_lower = message.lower()

        # Response states
        if any(word in message_lower for word in ['great', 'wonderful', 'excellent', 'amazing']):
            return "friendly_speaking"
        elif any(word in message_lower for word in ['sorry', 'concerned', 'worried', 'help']):
            return "empathetic"
        elif any(word in message_lower for word in ['goodbye', 'bye', 'see you']):
            return "farewell"
        else:
            return "neutral_speaking"

    async def handle_user_message(self, user_message: str) -> Dict:
        """Process user message and generate AI response"""
        # Change to listening state
        await self.broadcast({
            "type": "video_state",
            "state": "listening",
            "loop": self.video_states["listening"].get("loop", True)
        })

        # Get AI response from Ollama
        ollama_response = self.call_ollama(user_message)

        # Determine video state for response
        video_state = self.get_video_state_for_message(ollama_response)

        # Generate audio
        audio_path = self.tts_engine.synthesize(
            ollama_response,
            output_dir="outputs/audio"
        )

        # Send both audio and video state
        if audio_path:
            audio_url = f"/audio/{audio_path.split('/')[-1]}"
        else:
            audio_url = None

        return {
            "type": "ai_response",
            "audio_url": audio_url,
            "video_state": video_state,
            "text": ollama_response,
            "loop": self.video_states[video_state].get("loop", False)
        }

    def call_ollama(self, user_message: str) -> str:
        """Call Ollama API for AI response"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b-instruct-q4_K_M",
                    "prompt": f"You are Alice, a curious and empathetic guide in Cyberland. Respond helpfully and warmly to: {user_message}",
                    "stream": False,
                    "max_tokens": 150
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("response", "I'm sorry, I couldn't generate a response.")
        except Exception as e:
            print(f"Ollama API error: {e}")
            return "Connection error - please try again."

# Global state manager
state_manager = ChatStateManager()

# FastAPI app
app = FastAPI(title="Alice in Cyberland Chat")

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/audio", StaticFiles(directory="outputs/audio"), name="audio")
app.mount("/video", StaticFiles(directory="data/video_clips"), name="video")

# WebSocket endpoint
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    state_manager.active_clients.add(websocket)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "ai_response",
            "video_state": "greeting",
            "text": "Hello! I'm Alice, your guide to Cyberland. How can I help you today?",
            "audio_url": None,
            "loop": False
        })

        while True:
            # Receive user message
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if user_message.strip():
                # Process message and respond
                response = await state_manager.handle_user_message(user_message.strip())
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        state_manager.active_clients.remove(websocket)

# HTTP endpoint for health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "clients": len(state_manager.active_clients),
        "model": "llama3.1:8b-instruct-q4_K_M"
    }

# Serve chat interface
@app.get("/")
async def get_chat_interface():
    return HTMLResponse(open("static/index.html").read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.chat_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
