#!/usr/bin/env python3
"""
MPC Speech-to-Text Client
Client for interacting with the MPC Speech-to-Text service
"""

import asyncio
import json
import base64
import logging
from typing import Dict, Optional, Any, Callable
import aiohttp
import aiofiles
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MPCSpeechClient:
    """Client for MPC Speech-to-Text service"""
    
    def __init__(self, service_url: str = "http://localhost:8080", client_id: str = "claude_client"):
        self.service_url = service_url.rstrip('/')
        self.client_id = client_id
        self.auth_token: Optional[str] = None
        self.encryption_key: Optional[bytes] = None
        self.cipher: Optional[Fernet] = None
        self.session_id: Optional[str] = None
        
    async def authenticate(self) -> bool:
        """Authenticate with the MPC service"""
        try:
            async with aiohttp.ClientSession() as session:
                auth_data = {"client_id": self.client_id}
                
                async with session.post(
                    f"{self.service_url}/auth",
                    json=auth_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.auth_token = result["token"]
                        self.encryption_key = base64.b64decode(result["encryption_key"])
                        self.cipher = Fernet(self.encryption_key)
                        # logger.info("Authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def transcribe_file(self, audio_file_path: str, encrypt_response: bool = False) -> Dict[str, Any]:
        """Transcribe an audio file"""
        if not self.auth_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "X-Session-ID": self.session_id or "single_request"
            }
            
            if encrypt_response:
                headers["X-Encrypt-Response"] = "true"
            
            async with aiohttp.ClientSession() as session:
                with open(audio_file_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field('audio', f, filename=audio_file_path)
                    
                    async with session.post(
                        f"{self.service_url}/transcribe",
                        data=form_data,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            if encrypt_response and response.headers.get('X-Encrypted') == 'true':
                                # Handle encrypted response
                                encrypted_data = await response.read()
                                decrypted_data = self.cipher.decrypt(encrypted_data)
                                return json.loads(decrypted_data.decode())
                            else:
                                return await response.json()
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "error": f"HTTP {response.status}: {error_text}"
                            }
                            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def transcribe_bytes(self, audio_data: bytes, file_format: str = "wav", 
                              encrypt_response: bool = False) -> Dict[str, Any]:
        """Transcribe audio from bytes"""
        if not self.auth_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "X-Session-ID": self.session_id or "single_request"
            }
            
            if encrypt_response:
                headers["X-Encrypt-Response"] = "true"
            
            async with aiohttp.ClientSession() as session:
                form_data = aiohttp.FormData()
                form_data.add_field('audio', audio_data, filename=f'audio.{file_format}')
                
                async with session.post(
                    f"{self.service_url}/transcribe",
                    data=form_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        if encrypt_response and response.headers.get('X-Encrypted') == 'true':
                            encrypted_data = await response.read()
                            decrypted_data = self.cipher.decrypt(encrypted_data)
                            return json.loads(decrypted_data.decode())
                        else:
                            return await response.json()
                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def start_realtime_session(self, callback: Callable[[Dict], None] = None) -> 'RealtimeSession':
        """Start a real-time transcription session"""
        if not self.auth_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        return RealtimeSession(self.service_url, self.auth_token, callback)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.service_url}/status") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

class RealtimeSession:
    """Real-time transcription session using WebSocket"""
    
    def __init__(self, service_url: str, auth_token: str, callback: Optional[Callable] = None):
        self.service_url = service_url.replace('http', 'ws')
        self.auth_token = auth_token
        self.callback = callback
        self.websocket = None
        self.session_id = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to the WebSocket service"""
        try:
            session = aiohttp.ClientSession()
            self.websocket = await session.ws_connect(f"{self.service_url}/ws")
            
            # Authenticate
            auth_message = {
                "type": "auth",
                "token": self.auth_token
            }
            await self.websocket.send_text(json.dumps(auth_message))
            
            # Wait for auth response
            async for message in self.websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    if data.get('type') == 'auth_success':
                        self.session_id = data.get('session_id')
                        self.is_connected = True
                        logger.info(f"Connected with session ID: {self.session_id}")
                        break
                    elif data.get('type') == 'auth_error':
                        logger.error(f"Authentication failed: {data.get('message')}")
                        return False
                        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
        
        return True
    
    async def send_audio_chunk(self, audio_data: bytes, as_base64: bool = True):
        """Send audio chunk for transcription"""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        try:
            if as_base64:
                # Send as JSON with base64 encoded audio
                message = {
                    "type": "audio_chunk",
                    "audio_data": base64.b64encode(audio_data).decode()
                }
                await self.websocket.send_text(json.dumps(message))
            else:
                # Send as binary data
                await self.websocket.send_bytes(audio_data)
                
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
    
    async def listen_for_results(self):
        """Listen for transcription results"""
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        try:
            async for message in self.websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    
                    if data.get('type') == 'transcription_result':
                        result = data.get('data')
                        if self.callback:
                            self.callback(result)
                        else:
                            logger.info(f"Transcription result: {result}")
                            
                    elif data.get('type') == 'error':
                        logger.error(f"Service error: {data.get('message')}")
                        
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {self.websocket.exception()}')
                    break
                    
        except Exception as e:
            logger.error(f"Error listening for results: {e}")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False

# Example usage functions for Claude
async def transcribe_audio_file(file_path: str, service_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Helper function to transcribe an audio file"""
    client = MPCSpeechClient(service_url)
    
    # Authenticate
    if not await client.authenticate():
        return {"status": "error", "error": "Authentication failed"}
    
    # Transcribe
    result = await client.transcribe_file(file_path)
    return result

async def transcribe_audio_bytes(audio_data: bytes, service_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Helper function to transcribe audio from bytes"""
    client = MPCSpeechClient(service_url)
    
    # Authenticate
    if not await client.authenticate():
        return {"status": "error", "error": "Authentication failed"}
    
    # Transcribe
    result = await client.transcribe_bytes(audio_data)
    return result

async def check_service_status(service_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Helper function to check service status"""
    client = MPCSpeechClient(service_url)
    return await client.get_service_status()

# Example real-time transcription
async def realtime_transcription_example(service_url: str = "http://localhost:8080"):
    """Example of real-time transcription"""
    client = MPCSpeechClient(service_url)
    
    # Authenticate
    if not await client.authenticate():
        print("Authentication failed")
        return
    
    # Callback for handling results
    def handle_result(result):
        if result.get('status') == 'success':
            transcription = result.get('transcription', {})
            text = transcription.get('text', '')
            print(f"Transcribed: {text}")
        else:
            print(f"Error: {result.get('error')}")
    
    # Start real-time session
    session = await client.start_realtime_session(callback=handle_result)
    
    if await session.connect():
        print("Connected to real-time transcription service")
        
        # Example: Send audio chunks (you would replace this with actual audio data)
        # await session.send_audio_chunk(audio_chunk_bytes)
        
        # Listen for results
        await session.listen_for_results()
        
        # Close when done
        await session.close()

if __name__ == '__main__':
    # Example usage
    async def main():
        # Check service status
        status = await check_service_status()
        print(f"Service status: {status}")
        
        # Example transcription (replace with actual audio file path)
        # result = await transcribe_audio_file("/path/to/audio.wav")
        # print(f"Transcription result: {result}")
    
    asyncio.run(main())
