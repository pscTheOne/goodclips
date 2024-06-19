import os
import openai
import openai_key
import requests

# Set up your OpenAI API key
openai.api_key = openai_key.get_key()

# Directory containing audio files
audio_dir = "media"
# Output directory for text files
output_dir = "media"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to transcribe audio file using OpenAI API
def transcribe_audio_api(audio_path):
    audio_file = open(audio_path, "rb")
    response = openai.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1"
    )
    return response.text

# Process each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav") or filename.endswith(".mp3"):  # Add other formats if needed
        audio_path = os.path.join(audio_dir, filename)
        text = transcribe_audio_api(audio_path)

        # Save the transcribed text to a file
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, "w") as text_file:
            text_file.write(text)

        print(f"Transcribed {filename} and saved to {output_path}")

print("Transcription completed.")
