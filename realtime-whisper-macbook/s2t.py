import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor
model_name = "openai/whisper-small"  # or "medium", "large" depending on your GPU/memory
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Function to record audio
def record_audio(duration=5, fs=16000):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete")
    return recording.squeeze()

# Transcribe function
def transcribe(audio, sampling_rate=16000):
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model.generate(**inputs)
    transcription = processor.decode(logits[0], skip_special_tokens=True)
    return transcription

# Main loop
while True:
    audio = record_audio(duration=5)
    text = transcribe(audio)
    print(f"Transcribed Text: {text}")

