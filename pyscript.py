import os

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import csv
from transformers import pipeline

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
device = 0 if gpus else -1  # Use GPU if available, else use CPU

# Load the sentiment analysis model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=device)

# Read the Bible text
bible_verses = []
with open('bible_verses.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        bible_verses.append({'id': row['id'], 'b': row['b'], 'c': row['c'], 'v': row['v'], 'text': row['text']})


# Process the verses in batches
batch_size = 100
labeled_verses = []

for i in range(0, len(bible_verses), batch_size):
    batch = bible_verses[i:i+batch_size]
    texts = [verse['text'] for verse in batch]
    emotions = emotion_classifier(texts)
    
    # Print emotions to debug
    print(f"Batch {i // batch_size + 1}:")
    for emotion in emotions:
        print(emotion)

    for verse, emotion in zip(batch, emotions):
        if isinstance(emotion, list):
            top_emotion = max(emotion, key=lambda x: x['score'])['label']
        else:
            top_emotion = emotion['label']
        verse['emotion'] = top_emotion
        labeled_verses.append(verse)

# Save the labeled verses to a new CSV file
with open('labeled_bible_verses.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=['id', 'b', 'c', 'v', 'text', 'emotion'])
    writer.writeheader()
    writer.writerows(labeled_verses)
