import os
import re
import io
import pandas as pd
import matplotlib.pyplot as plt
from base64 import b64encode
from flask import Flask, render_template, request
from joblib import load
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure required NLTK data is downloaded
nltk.download("stopwords")
nltk.download("punkt")

app = Flask(__name__)

# Tokenize function
def tokenize(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    clean_tokens = [w for w in words if w not in stop_words]
    return clean_tokens

# Set up database connection using the relative path in the 'data/' folder
database_filepath = os.path.join(os.getcwd(), "data", "DisasterResponse.db")  # Absolute path for local and Render
engine = create_engine(f"sqlite:///{database_filepath}")

try:
    df = pd.read_sql_table("disaster_messages", engine)
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Lazy-load model from 'models/classifier.pkl'
model_filepath = os.path.abspath(os.path.join(os.getcwd(), "models", "classifier.pkl"))
model = None

@app.route("/index")
@app.route("/")
def index():
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    genre_plot = generate_plot(genre_names, genre_counts, "Message Genres", "Genre", "Count")
    category_plot = generate_plot(category_names, category_counts, "Message Categories", "Category", "Count")

    return render_template("master.html", genre_plot=genre_plot, category_plot=category_plot)

@app.route("/go")
def go():
    global model
    query = request.args.get("query", "")
    
    # Load model only when required
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

def generate_plot(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color="skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return plot_data

if __name__ == "__main__":
    app.run(debug=True)
