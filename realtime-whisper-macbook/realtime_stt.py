import torch
import sounddevice as sd
import numpy as np
from transformers import pipeline

# Choose the Whisper model (you can try "openai/whisper-medium" for potentially faster inference)
model_name = "openai/whisper-tiny"
#model_name = "openai/whisper-small"
#model_name = "openai/whisper-medium"
#model_name = "openai/whisper-large"
print("using model_name ", {model_name})
#device = "mps"  # You can try "mps" if you want to use Apple Silicon GPU (may require extra setup)
device = "cpu"  # You can try "mps" if you want to use Apple Silicon GPU (may require extra setup)
print("using device ", {device})


# Load the pipeline
recognizer = pipeline("automatic-speech-recognition", model=model_name, device=device)

# Audio parameters
samplerate = 16000
duration = 6  # Capture audio in chunks of 3 seconds

def callback(indata, frames, time, status):
    if status:
        print(f"Error in audio stream: {status}")
        return
    audio_chunk = indata.flatten().astype(np.float32)
    #print(f"Type of audio_chunk: {type(audio_chunk)}")
    if len(audio_chunk) > 0:
        try:
            forced_decoder_ids = recognizer.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")
            #print(f"Type of forced_decoder_ids: {type(forced_decoder_ids)}")
            #if forced_decoder_ids:
                #print(f"Sample of forced_decoder_ids: {forced_decoder_ids[:5]}")
            transcription = recognizer({"raw": audio_chunk, "sampling_rate": samplerate}, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
            #transcription = recognizer({"raw": audio_chunk, "input_features": audio_chunk, "sampling_rate": samplerate},
            #                            generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
            print(f"Transcribed (English): {transcription['text']}")
        except Exception as e:
            print(f"Error during transcription: {e}")

print("Listening... Press Ctrl+C to stop.")

try:
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback):
        while True:
            sd.sleep(int(duration * 1000))
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"Error during audio input: {e}")
