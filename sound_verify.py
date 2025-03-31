import torchaudio
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from speechbrain.pretrained import SpeakerRecognition
import time

# Load pre-trained model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="spkrec-ecapa-voxceleb"
)

# Function to record audio from microphone
def record_audio(filename, duration=3, samplerate=16000):
    print(f"🎤 Recording for {duration} seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wav.write(filename, samplerate, audio_data)  # Save as WAV file
    print(f"✅ Recording saved: {filename}")

# Speaker Verification Pipeline
reference_voice = None  # Stores the first recorded voice as reference

while True:
    user_input = input("\nPress ENTER to record (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        print("🔴 Stopping...")
        break

    # Record new voice input
    audio_filename = "current_speaker.wav"
    record_audio(audio_filename)

    # Load the recorded audio
    signal, fs = torchaudio.load(audio_filename)

    # If it's the first recording, set it as the reference
    if reference_voice is None:
        reference_voice = signal
        print("📌 First recording set as reference voice.")
        continue

    # Compare with the reference voice
    score = verification.verify_batch(reference_voice, signal)[0].item()
    threshold = 0.75  # Adjust based on your use case

    if score > threshold:
        print(f"✅ Same Speaker (Score: {score:.2f})")
    else:
        print(f"❌ Different Speaker (Score: {score:.2f})")
