import os
import json
from openai import OpenAI
import openai_key
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=openai_key.get_key())

# Set up your OpenAI API key

# Directory containing audio files
audio_dir = "media"

# Initialize the Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load sentences and vectors from JSON file
try:
    with open("sentences_with_vectors.json", "r") as json_file:
        data = json.load(json_file)
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit(1)

# Function to generate a response and find the closest matching sentence
def generate_response(question):
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are a art installation only capable of answering in short, funny sentences that sound like they could be movie quotes. your interaction with the user will be in something like natural spoken language. you are roleplaying as a telephone answering machine."},
      {"role": "user", "content": "what is your function?"}
    ],
    max_tokens=150)
    response_text = response.choices[0].message.content
    print("Response:", response_text)

    # Vectorize the response text
    response_vector = model.encode([response_text])[0]

    # Find the closest matching vector
    closest_match = None
    max_similarity = -1
    for entry in data:
        vectors = [vec["vector"] for vec in entry["sentences"]]
        similarities = cosine_similarity([response_vector], vectors)[0]
        max_idx = similarities.argmax()
        if similarities[max_idx] > max_similarity:
            max_similarity = similarities[max_idx]
            closest_match = {
                "wav_filename": entry["wav_filename"],
                "sentence": entry["sentences"][max_idx]["sentence"]
            }

    # Return the closest match information
    if closest_match:
        print(f"Closest match sentence: {closest_match['sentence']} Wavfile: {closest_match['wav_filename']}")
        return closest_match
    else:
        print("No close match found.")
        return None


# Example usage
question = "answer in 10 words or less, if you could would you marry?"
closest_match_info = generate_response(question)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)
