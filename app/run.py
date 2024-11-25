import os
import time
import pandas as pd
from flask import Flask, render_template, request
from joblib import load
from sqlalchemy import create_engine
import nltk
import gdown
import redis
import plotly
from plotly.graph_objs import Bar
import json

# Ensure required NLTK data is downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Flask app
app = Flask(__name__)

# Redis configuration
redis_url = os.environ.get("REDIS_URL")
if not redis_url:
    raise ValueError("REDIS_URL environment variable not set.")

# Connect to Redis
r = redis.StrictRedis.from_url(redis_url)

# Paths and configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
database_filepath = os.path.join(current_dir, "../data/DisasterResponse.db")
model_filepath = os.path.join(current_dir, "models", "classifier.pkl")
os.makedirs(os.path.dirname(model_filepath), exist_ok=True)

file_id = os.environ.get("FILE_ID")
if not file_id:
    raise ValueError("FILE_ID environment variable not set.")

# Set up database connection
engine = create_engine(f"sqlite:///{database_filepath}")
try:
    df = pd.read_sql_table("disaster_messages", engine)
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Function to check if the model is ready
def is_model_ready():
    return os.path.exists(model_filepath)

# Function to download the model
def download_model():
    """Download the model asynchronously."""
    if not os.path.exists(model_filepath):
        try:
            print("Starting model download...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filepath, quiet=False)
            r.set("model_downloaded", "True")  # Mark model as downloaded in Redis
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            r.set("model_downloaded", "False")  # Mark download failed in Redis

# Start model download as a background task
if not is_model_ready() and r.get("model_downloaded") is None:
    print("Model not found locally. Starting download...")
    r.set("model_downloaded", "False")  # Mark model as not downloaded in Redis
    download_model()

# Load the model when it's ready
model = None
if is_model_ready():
    try:
        model = load(model_filepath)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route("/")
@app.route("/index")
def index():
    """Render the main page with visuals."""
    if not is_model_ready():
        return "Model is initializing. Please refresh the page after a few moments."

    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"}
            }
        },
        {
            "data": [Bar(x=category_names, y=category_counts)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": -35}
            }
        }
    ]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("master.html", ids=ids, graphJSON=graphJSON)

@app.route("/go")
def go():
    """Handle user query and return classification results."""
    global model
    query = request.args.get("query", "")

    if model is None or not is_model_ready():
        return "Model is still downloading. Please try again later."

    try:
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))
    except Exception as e:
        print(f"Error in prediction: {e}")
        classification_results = {}

    return render_template("go.html", query=query, classification_result=classification_results)

if __name__ == "__main__":
    # Wait for model to be downloaded before starting the app
    while not is_model_ready() and r.get("model_downloaded") != b"True":
        print("Waiting for the model to be downloaded...")
        time.sleep(5)  # Check every 5 seconds

    if model is None:
        try:
            model = load(model_filepath)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
