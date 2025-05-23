#!/usr/bin/env python3
"""
MPC Speech-to-Text Service
Exposes realtime-whisper-macbook service through a secure MPC interface
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import hmac
import base64

import aiohttp
from aiohttp import web, WSMsgType
import aiofiles
import numpy as np
import soundfile as sf
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MPCSpeechService:
    """MPC Service for Speech-to-Text processing"""
    
    def __init__(self, whisper_service_url: str = "http://localhost:8000", 
                 encryption_key: Optional[bytes] = None):
        self.whisper_service_url = whisper_service_url
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.active_sessions: Dict[str, Dict] = {}
        self.auth_tokens: Dict[str, Dict] = {}
        
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return hashlib.sha256(f"{time.time()}{os.urandom(16)}".encode()).hexdigest()[:16]
    
    def generate_auth_token(self, client_id: str) -> str:
        """Generate JWT-like auth token for client"""
        payload = {
            "client_id": client_id,
            "issued_at": int(time.time()),
            "expires_at": int(time.time()) + 3600  # 1 hour expiry
        }
        token_data = json.dumps(payload).encode()
        signature = hmac.new(self.encryption_key, token_data, hashlib.sha256).hexdigest()
        token = base64.b64encode(token_data).decode() + "." + signature
        self.auth_tokens[token] = payload
        return token
    
    def verify_auth_token(self, token: str) -> Optional[Dict]:
        """Verify and decode auth token"""
        try:
            if "." not in token:
                return None
                
            token_b64, signature = token.rsplit(".", 1)
            token_data = base64.b64decode(token_b64.encode())
            
            # Verify signature
            expected_sig = hmac.new(self.encryption_key, token_data, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                return None
            
            payload = json.loads(token_data.decode())
            
            # Check expiry
            if payload.get("expires_at", 0) < int(time.time()):
                return None
                
            return payload
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None

    async def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data)
    
    async def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data)

    async def process_audio_chunk(self, audio_data: bytes, session_id: str, 
                                format: str = "wav") -> Dict[str, Any]:
        """Process audio chunk through whisper service"""
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Call the whisper service
                async with aiohttp.ClientSession() as session:
                    with open(temp_file_path, 'rb') as f:
                        form_data = aiohttp.FormData()
                        form_data.add_field('audio', f, filename=f'audio.{format}')
                        
                        async with session.post(
                            f"{self.whisper_service_url}/transcribe",
                            data=form_data
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return {
                                    "session_id": session_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "transcription": result,
                                    "status": "success"
                                }
                            else:
                                error_text = await response.text()
                                logger.error(f"Whisper service error: {error_text}")
                                return {
                                    "session_id": session_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "error": f"Service error: {response.status}",
                                    "status": "error"
                                }
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error"
            }

    async def handle_auth(self, request):
        """Handle authentication endpoint"""
        try:
            data = await request.json()
            client_id = data.get('client_id')
            
            if not client_id:
                return web.Response(
                    text=json.dumps({"error": "client_id required"}),
                    status=400,
                    content_type='application/json'
                )
            
            token = self.generate_auth_token(client_id)
            
            return web.Response(
                text=json.dumps({
                    "token": token,
                    "encryption_key": base64.b64encode(self.encryption_key).decode(),
                    "expires_in": 3600
                }),
                content_type='application/json'
            )
            
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return web.Response(
                text=json.dumps({"error": "Authentication failed"}),
                status=500,
                content_type='application/json'
            )

    async def handle_transcribe(self, request):
        """Handle transcription endpoint"""
        try:
            # Verify authentication
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return web.Response(
                    text=json.dumps({"error": "Missing or invalid authorization"}),
                    status=401,
                    content_type='application/json'
                )
            
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            auth_payload = self.verify_auth_token(token)
            if not auth_payload:
                return web.Response(
                    text=json.dumps({"error": "Invalid or expired token"}),
                    status=401,
                    content_type='application/json'
                )
            
            # Get session ID
            session_id = request.headers.get('X-Session-ID')
            if not session_id:
                session_id = self.generate_session_id()
            
            # Handle multipart form data (audio file)
            reader = await request.multipart()
            audio_data = None
            
            async for part in reader:
                if part.name == 'audio':
                    audio_data = await part.read()
                    break
            
            if not audio_data:
                return web.Response(
                    text=json.dumps({"error": "No audio data provided"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Process audio
            result = await self.process_audio_chunk(audio_data, session_id)
            
            # Encrypt sensitive data if requested
            if request.headers.get('X-Encrypt-Response') == 'true':
                encrypted_result = await self.encrypt_data(json.dumps(result).encode())
                return web.Response(
                    body=encrypted_result,
                    content_type='application/octet-stream',
                    headers={'X-Encrypted': 'true'}
                )
            
            return web.Response(
                text=json.dumps(result),
                content_type='application/json'
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return web.Response(
                text=json.dumps({"error": "Transcription failed"}),
                status=500,
                content_type='application/json'
            )

    async def handle_websocket(self, request):
        """Handle WebSocket connections for real-time transcription"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        session_id = None
        auth_payload = None
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        if data.get('type') == 'auth':
                            # Handle authentication
                            token = data.get('token')
                            auth_payload = self.verify_auth_token(token)
                            if auth_payload:
                                session_id = self.generate_session_id()
                                self.active_sessions[session_id] = {
                                    'client_id': auth_payload['client_id'],
                                    'connected_at': time.time(),
                                    'ws': ws
                                }
                                await ws.send_text(json.dumps({
                                    'type': 'auth_success',
                                    'session_id': session_id
                                }))
                            else:
                                await ws.send_text(json.dumps({
                                    'type': 'auth_error',
                                    'message': 'Invalid token'
                                }))
                                
                        elif data.get('type') == 'audio_chunk':
                            # Handle audio chunk
                            if not auth_payload or not session_id:
                                await ws.send_text(json.dumps({
                                    'type': 'error',
                                    'message': 'Not authenticated'
                                }))
                                continue
                            
                            # Decode base64 audio data
                            audio_b64 = data.get('audio_data', '')
                            audio_data = base64.b64decode(audio_b64)
                            
                            # Process audio
                            result = await self.process_audio_chunk(audio_data, session_id)
                            
                            await ws.send_text(json.dumps({
                                'type': 'transcription_result',
                                'data': result
                            }))
                            
                    except json.JSONDecodeError:
                        await ws.send_text(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        }))
                        
                elif msg.type == WSMsgType.BINARY:
                    # Handle binary audio data
                    if not auth_payload or not session_id:
                        await ws.send_text(json.dumps({
                            'type': 'error',
                            'message': 'Not authenticated'
                        }))
                        continue
                    
                    # Process binary audio data
                    result = await self.process_audio_chunk(msg.data, session_id)
                    
                    await ws.send_text(json.dumps({
                        'type': 'transcription_result',
                        'data': result
                    }))
                    
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up session
            if session_id and session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        return ws

    async def handle_status(self, request):
        """Handle status endpoint"""
        return web.Response(
            text=json.dumps({
                "service": "MPC Speech-to-Text",
                "status": "running",
                "active_sessions": len(self.active_sessions),
                "whisper_service": self.whisper_service_url,
                "timestamp": datetime.utcnow().isoformat()
            }),
            content_type='application/json'
        )

    def create_app(self) -> web.Application:
        """Create the web application"""
        app = web.Application()
        
        # Add routes
        app.router.add_post('/auth', self.handle_auth)
        app.router.add_post('/transcribe', self.handle_transcribe)
        app.router.add_get('/ws', self.handle_websocket)
        app.router.add_get('/status', self.handle_status)
        
        # Add CORS middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Session-ID, X-Encrypt-Response'
            return response
        
        app.middlewares.append(cors_middleware)
        
        return app

async def main():
    """Main function to run the MPC service"""
    # Configuration
    host = os.getenv('MPC_HOST', '0.0.0.0')
    port = int(os.getenv('MPC_PORT', '8080'))
    whisper_url = os.getenv('WHISPER_SERVICE_URL', 'http://localhost:8000')
    
    # Create service
    service = MPCSpeechService(whisper_service_url=whisper_url)
    app = service.create_app()
    
    logger.info(f"Starting MPC Speech-to-Text service on {host}:{port}")
    logger.info(f"Whisper service URL: {whisper_url}")
    logger.info(f"Encryption key: {base64.b64encode(service.encryption_key).decode()}")
    
    # Run the service
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    logger.info("MPC Service is running...")
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down MPC service...")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
