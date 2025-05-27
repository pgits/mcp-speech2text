#!/usr/bin/env python3
"""
HTTP API Wrapper for realtime-whisper-macbook
This adds HTTP endpoints to the existing whisper functionality
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import soundfile as sf
import torch
import whisper
from aiohttp import web
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperHTTPService:
    """HTTP wrapper for Whisper transcription service"""
    
    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        Initialize the Whisper HTTP service
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda, mps, auto)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = "cpu"  # Force CPU usage
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully")
        
        # Statistics
        self.stats = {
            "requests_processed": 0,
            "total_audio_duration": 0.0,
            "average_processing_time": 0.0,
            "start_time": time.time()
        }
    
    def transcribe_audio_file(self, audio_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_file_path: Path to audio file
            **kwargs: Additional Whisper parameters
            
        Returns:
            Transcription result dictionary
        """
        try:
            start_time = time.time()
            
            # Default Whisper options
            options = {
                "language": kwargs.get("language"),  # None for auto-detection
                "task": kwargs.get("task", "transcribe"),  # transcribe or translate
                "temperature": kwargs.get("temperature", 0.0),
                "best_of": kwargs.get("best_of", 5),
                "beam_size": kwargs.get("beam_size", 5),
                "patience": kwargs.get("patience", 1.0),
                "length_penalty": kwargs.get("length_penalty", 1.0),
                "suppress_tokens": kwargs.get("suppress_tokens", "-1"),
                "initial_prompt": kwargs.get("initial_prompt"),
                "condition_on_previous_text": kwargs.get("condition_on_previous_text", True),
                "fp16": kwargs.get("fp16", True if self.device == "cuda" else False),
                "compression_ratio_threshold": kwargs.get("compression_ratio_threshold", 2.4),
                "logprob_threshold": kwargs.get("logprob_threshold", -1.0),
                "no_speech_threshold": kwargs.get("no_speech_threshold", 0.6),
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Transcribe
            result = self.model.transcribe(audio_file_path, **options)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["requests_processed"] += 1
            if "segments" in result:
                audio_duration = max([seg["end"] for seg in result["segments"]], default=0)
                self.stats["total_audio_duration"] += audio_duration
            
            # Calculate average processing time
            total_requests = self.stats["requests_processed"]
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (total_requests - 1) + processing_time) / total_requests
            )
            
            # Add metadata
            result["processing_time"] = processing_time
            result["model"] = self.model_name
            result["device"] = self.device
            
            logger.info(f"Transcribed audio in {processing_time:.2f}s: '{result['text'][:100]}...'")
            
            return {
                "success": True,
                "result": result,
                "processing_time": processing_time,
                "model_info": {
                    "model": self.model_name,
                    "device": self.device
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_info": {
                    "model": self.model_name,
                    "device": self.device
                }
            }
    
    async def handle_transcribe(self, request):
        """Handle transcription HTTP requests"""
        try:
            # Handle multipart form data
            reader = await request.multipart()
            audio_data = None
            options = {}
            
            async for part in reader:
                if part.name == 'audio':
                    audio_data = await part.read()
                elif part.name == 'options':
                    options_text = await part.text()
                    try:
                        options = json.loads(options_text)
                    except json.JSONDecodeError:
                        pass
                elif part.name in ['language', 'task', 'temperature', 'beam_size']:
                    # Handle individual parameters
                    options[part.name] = await part.text()
            
            if not audio_data:
                return web.Response(
                    text=json.dumps({"error": "No audio data provided"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Convert options to appropriate types
                if 'temperature' in options:
                    options['temperature'] = float(options['temperature'])
                if 'beam_size' in options:
                    options['beam_size'] = int(options['beam_size'])
                
                # Transcribe
                result = self.transcribe_audio_file(temp_file_path, **options)
                
                return web.Response(
                    text=json.dumps(result),
                    content_type='application/json'
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return web.Response(
                text=json.dumps({"error": f"Request processing failed: {str(e)}"}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_health(self, request):
        """Health check endpoint"""
        uptime = time.time() - self.stats["start_time"]
        
        health_info = {
            "status": "healthy",
            "model": self.model_name,
            "device": self.device,
            "uptime_seconds": uptime,
            "statistics": self.stats.copy()
        }
        
        return web.Response(
            text=json.dumps(health_info),
            content_type='application/json'
        )
    
    async def handle_models(self, request):
        """List available models"""
        available_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        
        return web.Response(
            text=json.dumps({
                "available_models": available_models,
                "current_model": self.model_name,
                "device": self.device
            }),
            content_type='application/json'
        )
    
    def create_app(self) -> web.Application:
        """Create the web application"""
        app = web.Application(client_max_size=50*1024*1024)  # 50MB max file size
        
        # Add routes
        app.router.add_post('/transcribe', self.handle_transcribe)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/models', self.handle_models)
        
        # Add CORS middleware
        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == 'OPTIONS':
                # Handle preflight requests
                response = web.Response()
            else:
                response = await handler(request)
            
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
        
        app.middlewares.append(cors_middleware)
        
        return app

async def main():
    """Main function to run the Whisper HTTP service"""
    # Configuration from environment variables
    host = os.getenv('WHISPER_HOST', '127.0.0.1')
    port = int(os.getenv('WHISPER_PORT', '8000'))
    model_name = os.getenv('WHISPER_MODEL', 'base')
    device = os.getenv('WHISPER_DEVICE', 'auto')
    
    # Create service
    logger.info("Initializing Whisper HTTP service...")
    service = WhisperHTTPService(model_name=model_name, device=device)
    app = service.create_app()
    
    logger.info(f"Starting Whisper HTTP service on {host}:{port}")
    logger.info(f"Model: {model_name}, Device: {device}")
    
    # Run the service
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    logger.info("Whisper HTTP service is running!")
    logger.info(f"Endpoints available:")
    logger.info(f"  POST http://{host}:{port}/transcribe - Transcribe audio")
    logger.info(f"  GET  http://{host}:{port}/health - Health check")
    logger.info(f"  GET  http://{host}:{port}/models - List models")
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Whisper HTTP service...")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
