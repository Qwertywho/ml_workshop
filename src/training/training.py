import argparse
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from ..utils.utils import register_logger

# Set up custom logging
logger = logging.getLogger(__name__)
register_logger(logger)

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


@dataclass
class TrainingParams:
    """Dataclass for holding training parameters."""

    dataset_name: str = "imdb"
    output_dir: str = "./results"
    num_train_epochs: int = 2
    batch_size: int = 8
    subset_size: int = 20000
    eval_subset_size: int = 5000
    logging_steps: int = 100
    max_length: int = 512
    device: str = "cpu"
    hidden_size: int = 128  # Size of the hidden layer for MLP
    input_size: int = 5000  # Maximum features for TF-IDF
    output_size: int = 2  # Binary classification


class MLPTuner:
    def __init__(self, params: TrainingParams):
        self.params = params
        self.model = None
        self.vectorizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None

    def load_data_and_vectorizer(self):
        """Load dataset and apply TF-IDF vectorization."""
        dataset = load_dataset(self.params.dataset_name)

        train_texts = dataset["train"]["text"][: self.params.subset_size]
        train_labels = torch.tensor(
            dataset["train"]["label"][: self.params.subset_size]
        )

        test_texts = dataset["test"]["text"][: self.params.eval_subset_size]
        test_labels = torch.tensor(
            dataset["test"]["label"][: self.params.eval_subset_size]
        )

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=self.params.input_size)
        X_train = self.vectorizer.fit_transform(train_texts).toarray()
        X_test = self.vectorizer.transform(test_texts).toarray()

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # Create DataLoader
        self.train_loader = DataLoader(
            TensorDataset(X_train_tensor, train_labels),
            batch_size=self.params.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test_tensor, test_labels),
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def build_model(self):
        """Build the MLP model."""
        self.model = MLP(
            input_size=self.params.input_size,
            hidden_size=self.params.hidden_size,
            output_size=self.params.output_size,
        ).to(self.params.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        """Train the MLP model."""
        self.model.train()
        logger.info("start training")
        for epoch in range(self.params.num_train_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.params.device), labels.to(
                    self.params.device
                )

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if (i + 1) % self.params.logging_steps == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.params.num_train_epochs}], Step [{i+1}], Loss: {loss.item():.4f}"
                    )

            logger.info(
                f"Epoch [{epoch+1}/{self.params.num_train_epochs}], Loss: {running_loss/len(self.train_loader):.4f}"
            )

    def evaluate(self):
        """Evaluate the MLP model."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.params.device), labels.to(
                    self.params.device
                )
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Test Accuracy: {accuracy:.2f}%")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an MLP for sentiment analysis.")

    parser.add_argument(
        "--dataset_name", type=str, default="imdb", help="Name of the dataset to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the model and results.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=20000,
        help="Number of samples to use for training.",
    )
    parser.add_argument(
        "--eval_subset_size",
        type=int,
        default=5000,
        help="Number of samples to use for evaluation.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Number of steps for logging."
    )
    parser.add_argument(
        "--input_size", type=int, default=5000, help="Number of features for TF-IDF."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Number of neurons in the hidden layer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Ensure that the script runs on CPU if no GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize training parameters
    params = TrainingParams(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        eval_subset_size=args.eval_subset_size,
        logging_steps=args.logging_steps,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        device=str(device),
    )

    # Create an instance of the MLP fine-tuner class
    mlp_tuner = MLPTuner(params)

    # Load the data and build the vectorizer
    mlp_tuner.load_data_and_vectorizer()

    # Build the MLP model
    mlp_tuner.build_model()

    # Train the model
    mlp_tuner.train()

    # Evaluate the model
    mlp_tuner.evaluate()
