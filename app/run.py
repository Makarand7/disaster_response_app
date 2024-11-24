import os
import re
import io
import pandas as pd
from base64 import b64encode
from flask import Flask, render_template, request
from joblib import load
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import gdown
from celery import Celery

import plotly
from plotly.graph_objs import Bar
import json

# Ensure required NLTK data is downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Celery configuration
redis_url = os.environ.get("REDIS_URL")  # Use environment variable for Redis URL
if not redis_url:
    raise ValueError("REDIS_URL environment variable not set.")

# Setting the Celery broker and backend
celery = Celery(
    "run",  # Name of the current Flask app (this will be the Celery worker name)
    broker=redis_url,  # Redis URL fetched from the environment
    backend=None  # No results stored to avoid exceeding the limit
)

app = Flask(__name__)

# Tokenize function
def tokenize(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    clean_tokens = [w for w in words if w not in stop_words]
    return clean_tokens

# Resolve database path
current_dir = os.path.dirname(os.path.abspath(__file__))
database_filepath = os.path.join(current_dir, "../data/DisasterResponse.db")

# Set up database connection
engine = create_engine(f"sqlite:///{database_filepath}")

try:
    df = pd.read_sql_table("disaster_messages", engine)
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Replace hardcoded file_id with environment variable
file_id = os.environ.get("FILE_ID")  # Environment variable for Google Drive model file ID
if not file_id:
    raise ValueError("FILE_ID environment variable not set.")

# Define the path to save the model
model_filepath = os.path.join(current_dir, "models", "classifier.pkl")  # Use a specific directory for model

# Download model if not already present (Celery task will handle background download)
if not os.path.exists(model_filepath):
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Trigger background task to download the model via Celery
    @celery.task
    def download_model():
        if not os.path.exists(model_filepath):
            try:
                # Ensure the model is downloaded to the correct directory
                gdown.download(download_url, model_filepath, quiet=False)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")

    # Trigger the Celery task
    download_model.apply_async()

# Load the model for later use (ensure download task is complete)
model = None
try:
    # Make sure that the model exists before loading
    if os.path.exists(model_filepath):
        model = load(model_filepath)
        print("Model loaded successfully.")
    else:
        raise Exception(f"Model file not found: {model_filepath}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/index")
@app.route("/")
def index():
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category", 'tickangle': -35}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route("/go")
def go():
    global model
    query = request.args.get("query", "")

    if model is None:
        try:
            model = load(model_filepath)
        except Exception as e:
            print(f"Error loading model: {e}")
            return render_template("go.html", query=query, classification_result={})

    try:
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))
    except Exception as e:
        print(f"Error in prediction: {e}")
        classification_results = {}

    return render_template("go.html", query=query, classification_result=classification_results)

if __name__ == "__main__":
    # Get the port from the environment variable; default to 5000 if not found
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 for external connections
    app.run(host="0.0.0.0", port=port, debug=True)
