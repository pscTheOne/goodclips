import os
import nltk
import json
from sentence_transformers import SentenceTransformer
import torch
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Directory containing text files
text_dir = "media"

# Dictionary to hold sentences with unique filenames
data = {}

# Function to read text files and extract sentences
def load_sentences(text_dir):
    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):  # Ensure only text files are processed
            text_path = os.path.join(text_dir, filename)
            with open(text_path, "r") as text_file:
                text = text_file.read()
                sentences = sent_tokenize(text)
                wav_filename = os.path.splitext(filename)[0] + ".wav"
                if wav_filename not in data:
                    data[wav_filename] = []
                data[wav_filename].extend(sentences)

# Load sentences from text files
load_sentences(text_dir)

# Initialize the Sentence-BERT model
try:
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Convert the dictionary to a list of dictionaries with vectors
output_data = []
for wav_filename, sentences in data.items():
    try:
        sentence_vectors = model.encode(sentences)
        sentences_with_vectors = []
        for i, sentence in enumerate(sentences):
            sentences_with_vectors.append({
                "sentence": sentence,
                "vector": sentence_vectors[i].tolist()  # Convert numpy array to list for JSON serialization
            })
        output_data.append({
            "wav_filename": wav_filename,
            "sentences": sentences_with_vectors
        })
    except Exception as e:
        print(f"Error encoding sentences for {wav_filename}: {e}")

# Save sentences and their vectors to a JSON file
try:
    with open("sentences_with_vectors.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)
    print("Sentences and vectors extracted and saved to JSON.")
except Exception as e:
    print(f"Error saving to JSON file: {e}")
