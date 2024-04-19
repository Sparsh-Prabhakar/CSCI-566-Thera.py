import sounddevice as sd
from scipy.io.wavfile import write
import openai
from config import *
from src.getGPT import *

def record_audio(filename, duration=5, fs=44100):
    print("Recording...")
    try:
        # Record from the microphone with mono output
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is complete
        write(filename, fs, recording)  # Save the recording in WAV format
        print(f"Finished recording, saved as {filename}")
    except Exception as e:
        print(f"An error occurred during recording: {e}")

client = azure_client #openai.OpenAI(api_key= api_key) 
def transcribe_audio(file_path):
    record_audio(file_path,duration=5)
    with open("output.wav", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcription.text
if __name__=='__main__':
    print(transcribe_audio('output.wav'))