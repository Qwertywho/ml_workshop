from locust import HttpUser, task, between
from datasets import load_dataset
import random

class IMDBUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between task executions

    # Load the IMDB dataset (unsupervised subset)
    dataset = load_dataset("imdb", split="unsupervised")

    def on_start(self):
        """
        Called when a Locust instance starts before any task is executed.
        Shuffle the dataset to avoid querying in the same order every time.
        """
        self.text_samples = [sample['text'] for sample in self.dataset]
        random.shuffle(self.text_samples)

    @task
    def query_predict_endpoint(self):
        """
        Task to query the /predict endpoint.
        """
        # Randomly select a text sample from the dataset
        random_text = random.choice(self.text_samples)

        # JSON payload with "text" as the key
        json_payload = {
            "text": random_text
        }

        # Send a POST request to the /predict endpoint
        self.client.post("/predict", json=json_payload)
