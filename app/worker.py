from celery import Celery
import os
import gdown

# Celery configuration
celery = Celery(
    "worker",
    backend="redis://localhost:6379/0",
    broker="redis://localhost:6379/0"
)

# Model download task
@celery.task
def download_model():
    file_id = os.environ.get("GOOGLE_DRIVE_MODEL_FILE_ID", "1eMAjZM3_oCC_cV-EVUswCnL3_jj31ryH")
    model_filepath = os.path.abspath(os.path.join(os.getcwd(), "../models/classifier.pkl"))
    download_url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(model_filepath):
        try:
            gdown.download(download_url, model_filepath, quiet=False)
            return "Model downloaded successfully."
        except Exception as e:
            return f"Error downloading model: {e}"
    return "Model already exists."

