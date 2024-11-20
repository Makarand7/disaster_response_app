import json
import pandas as pd
import os
import gdown
import re
import matplotlib.pyplot as plt
import io
from base64 import b64encode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from joblib import load
from sqlalchemy import create_engine
import nltk

# Ensure required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Define the tokenize function
def tokenize(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    clean_tokens = [w for w in words if w not in stop_words]
    return clean_tokens

# Set up database connection (SQLite for local DB)
database_filepath = os.path.abspath(os.path.join(os.getcwd(), 'data/DisasterResponse.db'))
database_url = f'sqlite:///{database_filepath}'

# Connect to the database using SQLAlchemy (SQLite in this case)
try:
    engine = create_engine(database_url)
    # Ensure the 'disaster_messages' table exists in your SQLite database
    df = pd.read_sql_table('disaster_messages', engine)
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit(1)

# Google Drive model file ID
file_id = os.environ.get('GOOGLE_DRIVE_MODEL_FILE_ID', '1eMAjZM3_oCC_cV-EVUswCnL3_jj31ryH')  # Default ID if not set
model_filepath = os.path.abspath(os.path.join(os.getcwd(), 'models/classifier.pkl'))
download_url = f'https://drive.google.com/uc?id={file_id}'

def download_model():
    if not os.path.exists(model_filepath):
        try:
            gdown.download(download_url, model_filepath, quiet=False)
        except Exception as e:
            print(f"Error downloading the model: {e}")
            exit(1)

download_model()

# Load the pre-trained model
try:
    model = load(model_filepath)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Generate plot function using Matplotlib
def generate_plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return plot_data

@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    genre_plot = generate_plot(genre_names, genre_counts, "Distribution of Message Genres", "Genre", "Count")
    category_plot = generate_plot(category_names, category_counts, "Distribution of Message Categories", "Category", "Count")

    return render_template('master.html', genre_plot=genre_plot, category_plot=category_plot)

@app.route('/go')
def go():
    query = request.args.get('query', '')
    try:
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))
    except Exception as e:
        print(f"Error in prediction: {e}")
        classification_results = {}

    return render_template('go.html', query=query, classification_result=classification_results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no port is specified
    app.run(host='0.0.0.0', port=port, debug=True)
