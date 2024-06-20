import os
import openai
import openai_key
import time
from requests.exceptions import RequestException

# Set up your OpenAI API key
openai.api_key = openai_key.get_key()

# Directory containing audio files
audio_dir = "media"
# Output directory for text files
output_dir = "media"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to transcribe audio file using OpenAI API with timeout and retries
def transcribe_audio_api(audio_path, retries=3, timeout=60):
    attempt = 0
    while attempt < retries:
        try:
            audio_file = open(audio_path, "rb")
            response = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                timeout=timeout
            )
            return response.text
        except (openai.error.OpenAIError, RequestException) as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    raise Exception(f"Failed to transcribe {audio_path} after {retries} attempts")

# Process each audio file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav") or filename.endswith(".mp3"):  # Add other formats if needed
        audio_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")

        # Skip if the transcription file already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename} as transcription already exists.")
            continue

        try:
            text = transcribe_audio_api(audio_path)

            # Save the transcribed text to a file
            with open(output_path, "w") as text_file:
                text_file.write(text)

            print(f"Transcribed {filename} and saved to {output_path}")
        except Exception as e:
            print(f"Failed to transcribe {filename}: {e}")

print("Transcription completed.")
