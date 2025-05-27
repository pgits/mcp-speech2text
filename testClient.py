#!/usr/bin/env python3
"""
Test client for MPC Speech-to-Text service
"""

import asyncio
from mpc_speech_client import transcribe_audio_file, check_service_status

async def main():
    """Main test function"""
    print("Testing MPC Speech-to-Text Client")
    print("=" * 40)
    
    # First, check if the service is running
    print("1. Checking service status...")
    status = await check_service_status()
    print(f"Service status: {status}")
    
    if status.get("status") == "error":
        print("⚠️  Service appears to be down. Make sure to start the services first:")
        print("   1. Start Whisper HTTP service: python whisper_http_wrapper.py")
        print("   2. Start MPC service: python mpc_speech_service.py")
        return
    
    print("\n2. Testing audio file transcription...")
    
    # Test with an audio file (you'll need to provide a real audio file)
    audio_file = "audio.wav"  # Replace with your actual audio file path
    
    try:
        result = await transcribe_audio_file(audio_file)
        print(f"Transcription result: {result}")
        
        if result.get("status") == "success":
            transcription = result.get("transcription", {})
            text = transcription.get("text", "No text found")
            print(f"✅ Transcribed text: '{text}'")
        else:
            print(f"❌ Transcription failed: {result.get('error')}")
            
    except FileNotFoundError:
        print(f"❌ Audio file '{audio_file}' not found.")
        print("   Please provide a valid audio file path or create a test audio file.")
    except Exception as e:
        print(f"❌ Error during transcription: {e}")

if __name__ == "__main__":
    asyncio.run(main())