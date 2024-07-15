from flask import Flask, request, render_template
from transformers import pipeline
import csv
import random

app = Flask(__name__)

# Load the sentiment analysis model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load labeled Bible verses from CSV
labeled_verses = []

with open('labeled_bible_verses.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        labeled_verses.append({
            'id': row['id'],
            'b': row['b'],
            'c': row['c'],
            'v': row['v'],
            'text': row['text'],
            'emotion': row['emotion']
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_verse', methods=['POST'])
def get_verse():
    user_input = request.form['feeling']
    emotions = emotion_classifier(user_input)

    emotion_labels = [emotion['label'].lower() for emotion in emotions]
    selected_verse = None

    # Select a random verse based on detected emotion
    for label in emotion_labels:
        matching_verses = [verse['text'] for verse in labeled_verses if verse['emotion'] == label]
        if matching_verses:
            selected_verse = random.choice(matching_verses)
            break

    return render_template('result.html', user_input=user_input, selected_verse=selected_verse)

if __name__ == '__main__':
    app.run(debug=True)
