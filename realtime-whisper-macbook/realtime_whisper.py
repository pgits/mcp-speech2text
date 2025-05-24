import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import pipeline
import time
import threading
import queue
import warnings
# Filter the specific FutureWarning from Hugging Face Transformers
warnings.filterwarnings(
    "ignore",
    message="The input name `inputs` is deprecated\. Please make sure to use `input_features` instead\.",
    category=FutureWarning,
    module="transformers" # This targets the warning specifically from the transformers module
)


# --- Configuration ---
MODEL_NAME = "openai/whisper-tiny.en" # "tiny.en" for English-only, "base.en", "small.en", etc.
                                   # Smaller models are faster but less accurate.
SAMPLE_RATE = 16000 # Whisper models expect 16kHz audio
CHUNK_SIZE_SECONDS = 5 # Process audio in 5-second chunks
AUDIO_BUFFER_SIZE_SECONDS = 15 # Keep a buffer of recent audio to avoid cutting off sentences

# --- Initialize Whisper Pipeline ---
# Use the 'mps' device for Apple Silicon (M1/M2/M3 chips) for faster processing
# Fallback to 'cpu' if 'mps' is not available or if you're on an older Intel Mac
try:
    pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device="mps")
    print(f"Using MPS for model: {MODEL_NAME}")
except Exception:
    pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device="cpu")
    print(f"Using CPU for model: {MODEL_NAME} (MPS not available or failed)")

# --- Audio Recording Setup ---
q = queue.Queue()
recording = False
current_audio_buffer = np.array([], dtype=np.int16)
transcription_thread = None
buffer_lock = threading.Lock()

def callback(indata, frames, time, status):
    """This is called (continuously) for each audio block."""
    if status:
        print(status)
    if recording:
        q.put(indata.copy())

def process_audio():
    global current_audio_buffer
    while recording or not q.empty():
        try:
            data = q.get(timeout=1) # Get audio data from the queue
            # Ensure data is 16-bit PCM
            if data.dtype == np.float32:
                data = (data * 32767).astype(np.int16)
            elif data.dtype != np.int16:
                # Attempt to convert to 16-bit if not already
                data = data.astype(np.int16)

            with buffer_lock:
                current_audio_buffer = np.append(current_audio_buffer, data.flatten())

            # Only keep the last AUDIO_BUFFER_SIZE_SECONDS of audio
            max_samples = int(AUDIO_BUFFER_SIZE_SECONDS * SAMPLE_RATE)
            if len(current_audio_buffer) > max_samples:
                current_audio_buffer = current_audio_buffer[-max_samples:]

            # Process a chunk for transcription
            chunk_samples = int(CHUNK_SIZE_SECONDS * SAMPLE_RATE)
            if len(current_audio_buffer) >= chunk_samples:
                # Take the most recent chunk for processing
                audio_chunk = current_audio_buffer[-chunk_samples:]

                # Save a temporary WAV file for the pipeline (required by some versions or configurations)
                # A more optimized approach would be to pass the numpy array directly if the pipeline supports it
                temp_wav_path = "temp_audio_chunk.wav"
                wavfile.write(temp_wav_path, SAMPLE_RATE, audio_chunk)

                # Transcribe the audio chunk
                try:
                    start_time = time.time()
                    transcription = pipe(temp_wav_path)["text"]
                    #transcription = pipe(temp_wav_path, generate_kwargs={"task": "transcribe", "language": "english"})["text"]
                    end_time = time.time()
                    print(f"[{transcription.strip()}] (processed in {end_time - start_time:.2f}s)")
                    # Clear the processed chunk from the buffer to avoid re-processing
                    current_audio_buffer = current_audio_buffer[:-chunk_samples]
                except Exception as e:
                    print(f"Error during transcription: {e}")
                finally:
                    # Clean up temporary file (optional if you don't mind it)
                    import os
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in process_audio: {e}")

print(f"Starting real-time speech to text using {MODEL_NAME}...")
print("Speak into your microphone. Press Enter to stop.")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype='float32'):
        recording = True
        transcription_thread = threading.Thread(target=process_audio)
        transcription_thread.daemon = True # Allow program to exit even if thread is running
        transcription_thread.start()
        input() # This will block until you press Enter

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    recording = False
    if transcription_thread and transcription_thread.is_alive():
        q.put(None) # Signal the thread to stop if it's waiting
        transcription_thread.join(timeout=5) # Wait for the thread to finish
        if transcription_thread.is_alive():
            print("Transcription thread did not terminate cleanly.")
    print("Stopped recording and transcription.")
