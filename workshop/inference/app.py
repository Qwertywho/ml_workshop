import argparse
import os

import joblib
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from workshop.utils.model import MLP

# Define FastAPI app
app = FastAPI()

# Define input data schema
class InputText(BaseModel):
    text: str


# Prometheus metrics
REQUEST_COUNT = Counter("app_request_count", "Total number of requests")
RESPONSE_TIME = Histogram("app_response_time_seconds", "Response time in seconds")


class ModelAPI:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self._load_model_and_vectorizer()

        # Initialize FastAPI app within the class
        self.app = FastAPI()

        # Add endpoint within the class
        @self.app.post("/predict")
        @RESPONSE_TIME.time()  # Prometheus histogram to measure response time
        async def get_prediction(input_text: InputText):
            REQUEST_COUNT.inc()  # Increment request count
            return self.predict(input_text.text)

        # Add a Prometheus metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return JSONResponse(
                content=generate_latest(), media_type=CONTENT_TYPE_LATEST
            )

    def _load_model_and_vectorizer(self):
        """Load model and vectorizer from the provided directory."""
        model_path = os.path.join(self.model_dir, "pytorch_model.bin")
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Model or vectorizer files not found in the specified directory."
            )

        # Load model
        self.model = MLP(input_size=5000, hidden_size=128, output_size=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load vectorizer
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str):
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


# Main function to accept arguments and run the app
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Run FastAPI to serve a trained MLP model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model weights and vectorizer",
    )
    args = parser.parse_args()

    # Instantiate the ModelAPI class with the given model directory
    model_api = ModelAPI(model_dir=args.model_dir)

    # Run the FastAPI app from the class
    uvicorn.run(model_api.app, host="0.0.0.0", port=8000, reload=True)
