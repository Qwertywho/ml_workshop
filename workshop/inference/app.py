import argparse
import os

import joblib
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from huggingface_hub import hf_hub_download
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# from prometheus_client import make_asgi_app
from pydantic import BaseModel

from workshop.utils.model import MLP

parser = argparse.ArgumentParser(description="Run FastAPI to serve a trained MLP model")
parser.add_argument(
    "--repo_id",
    type=str,
    required=True,
    help="Directory containing model weights and vectorizer",
)
parser.add_argument(
    "--port",
    type=int,
    required=False,
    help="Port for running the app",
    default=8000,
)
parser.add_argument(
    "--model_input_size",
    type=int,
    default=5000,
    help="Number of features for TF-IDF.",
)
parser.add_argument(
    "--model_hidden_size",
    type=int,
    default=128,
    help="Number of neurons in the hidden layer.",
)
args = parser.parse_args()


class InputText(BaseModel):
    text: str


# Prometheus metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests")
RESPONSE_TIME = Histogram("app_response_time_seconds", "Response time in seconds")


class ModelAPI:
    def __init__(self, repo_id: str, model_input_size: int, model_hidden_size: int):
        self.repo_id = repo_id
        self.model = None
        self.vectorizer = None
        self._load_model_and_vectorizer(
            model_input_size=model_input_size, model_hidden_size=model_hidden_size
        )

        # Initialize FastAPI app within the class
        self.app = FastAPI()

        # Add endpoint within the class

    def _load_model_and_vectorizer(self, model_input_size: int, model_hidden_size: int):
        """Load model and vectorizer from the huggingface hub"""
        model_path = hf_hub_download(repo_id=self.repo_id, filename="pytorch_model.bin")
        vectorizer_path = hf_hub_download(self.repo_id, "tfidf_vectorizer.pkl")
        print(vectorizer_path)

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Model or vectorizer files not found in the specified directory."
            )

        # Load model
        self.model = MLP(
            input_size=model_input_size, hidden_size=model_hidden_size, output_size=2
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        # Load vectorizer
        self.vectorizer = joblib.load(vectorizer_path)

    async def predict(self, text: str):
        """Predict the class of the input text."""
        if self.model is None or self.vectorizer is None:
            raise HTTPException(
                status_code=500, detail="Model or vectorizer not loaded."
            )

        # Transform input text using the loaded vectorizer
        features = self.vectorizer.transform([text]).toarray()
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            output = self.model(features_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        return {"prediction": predicted_class}


model_api = ModelAPI(
    repo_id=args.repo_id,
    model_input_size=args.model_input_size,
    model_hidden_size=args.model_hidden_size,
)

# Define FastAPI app
app = FastAPI()


@app.post("/predict")
@RESPONSE_TIME.time()  # Prometheus histogram to measure response time
async def get_prediction(input_text: InputText):
    """Prediction using the model trained"""
    REQUEST_COUNT.inc()  # Increment request count
    output_json = await model_api.predict(input_text.text)
    return JSONResponse(output_json)


# Add a Prometheus metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Endpoint for metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Run the FastAPI app from the class
    uvicorn.run("__main__:app", host="0.0.0.0", port=args.port, reload=True)
