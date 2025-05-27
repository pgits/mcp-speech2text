#!/usr/bin/env python3
"""
Real-time speech-to-text client using microphone input
"""

import asyncio
import queue
import tempfile
import wave
import numpy as np
import sounddevice as sd
from mpc_speech_client import transcribe_audio_file, check_service_status

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
CHUNK_DURATION = 5  # Back to 5 seconds for better stability
SILENCE_THRESHOLD = 0.01  # Back to original threshold
BUFFER_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Back to full chunk duration

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.buffer = np.array([], dtype=DTYPE)

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"⚠️  Status: {status}")
        
        # Convert to float for RMS calculation
        audio_data = indata.copy()
        audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Calculate RMS to detect silence
        rms = np.sqrt(np.mean(audio_float**2))
        
        if rms > SILENCE_THRESHOLD:
            # Add to buffer
            self.buffer = np.append(self.buffer, audio_data.flatten())
            
            # If buffer is large enough, process it
            if len(self.buffer) >= BUFFER_SIZE:
                self.audio_queue.put(self.buffer.reshape(-1, 1))
                self.buffer = np.array([], dtype=DTYPE)

    async def process_audio_chunk(self):
        """Process accumulated audio data"""
        if self.audio_queue.empty():
            return None
        
        # Get the next chunk from the queue
        audio_data = self.audio_queue.get()
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())
            
            # Transcribe the audio
            result = await transcribe_audio_file(temp_file.name)
            
            if result.get("status") == "success":
                transcription = result.get("transcription", {})
                if isinstance(transcription, dict):
                    result_data = transcription.get("result", {})
                    text = result_data.get("text", "").strip()
                    if text:
                        print(f"🎤 You said: '{text}'")
            else:
                print(f"❌ Transcription failed: {result.get('error')}")

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * 0.2),  # Back to 200ms blocks
            callback=self.audio_callback
        )
        self.stream.start()
        print("🎤 Started recording... (Press Ctrl+C to stop)")

    def stop_recording(self):
        """Stop recording audio"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False

async def main():
    """Main function"""
    print("Real-time Speech-to-Text Client")
    print("=" * 40)
    
    # Check if the service is running
    print("Checking service status...")
    status = await check_service_status()
    
    if status.get("status") == "error":
        print("⚠️  Service appears to be down. Make sure to start the services first:")
        print("   1. Start Whisper HTTP service: python whisper_http_wrapper.py")
        print("   2. Start MPC service: python mpc_speech_service.py")
        return
    
    print("✅ Service is running!")
    print("\nStarting real-time speech recognition...")
    print("Speak into your microphone. Your speech will be transcribed immediately.")
    print("Press Ctrl+C to stop.")
    
    processor = AudioProcessor()
    processor.start_recording()
    
    try:
        while processor.is_recording:
            await processor.process_audio_chunk()
            await asyncio.sleep(0.05)  # Reduced sleep time for more frequent processing
    except KeyboardInterrupt:
        print("\n👋 Stopping recording...")
    finally:
        processor.stop_recording()

if __name__ == "__main__":
    asyncio.run(main()) 