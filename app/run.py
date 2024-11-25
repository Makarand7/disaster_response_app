import os
import re
import time
import pandas as pd
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
redis_url = os.environ.get("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL environment variable not set.")

celery = Celery(
    "run",
    broker=redis_url,
    backend=None
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

# Google Drive file ID from environment variable
file_id = os.environ.get("FILE_ID")
if not file_id:
    raise ValueError("FILE_ID environment variable not set.")

# Define the model path
model_filepath = os.path.join(current_dir, "models", "classifier.pkl")
os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

# Celery task for downloading the model
@celery.task
def download_model():
    if not os.path.exists(model_filepath):
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filepath, quiet=False)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")

# Ensure the model is downloaded before starting the app
if not os.path.exists(model_filepath):
    print("Model not found locally. Starting download via Celery...")
    download_model.apply_async()

    # Wait for the download to complete
    print("Waiting for the model to be downloaded...")
    while not os.path.exists(model_filepath):
        time.sleep(5)  # Check every 5 seconds
    print("Model download complete.")

# Load the model
model = None
try:
    model = load(model_filepath)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
@app.route("/index")
def index():
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Create visuals
    graphs = [
        {
            "data": [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"}
            }
        },
        {
            "data": [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": -35}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("master.html", ids=ids, graphJSON=graphJSON)

@app.route("/go")
def go():
    global model
    query = request.args.get("query", "")

    if model is None:
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
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
