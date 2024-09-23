import argparse
import logging
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Set up custom logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    """Dataclass for holding training parameters."""

    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    output_dir: str = "./results"
    num_train_epochs: int = 2
    batch_size: int = 8
    subset_size: int = 20000
    eval_subset_size: int = 5000
    logging_steps: int = 100
    max_length: int = 512
    device: str = "cpu"


class CustomLoggerCallback(TrainerCallback):
    """Custom callback class to log training information."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log the information using the custom logger."""
        if logs is not None:
            logger.info(f"Step: {state.global_step}, Logs: {logs}")


class DistilBertFineTuner:
    def __init__(self, params: TrainingParams):
        self.params = params
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.params.model_name, num_labels=2
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.params.model_name)

    def load_and_tokenize_data(self):
        """Load dataset and tokenize the text."""
        dataset = load_dataset(self.params.dataset_name)

        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.params.max_length,
            )

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        self.train_dataset = (
            tokenized_dataset["train"]
            .shuffle(seed=42)
            .select(range(self.params.subset_size))
        )
        self.test_dataset = (
            tokenized_dataset["test"]
            .shuffle(seed=42)
            .select(range(self.params.eval_subset_size))
        )

    def train(self):
        """Set training arguments and train the model."""
        training_args = TrainingArguments(
            output_dir=self.params.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.params.batch_size,
            per_device_eval_batch_size=self.params.batch_size,
            num_train_epochs=self.params.num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=self.params.logging_steps,
            load_best_model_at_end=True,
            report_to="none",  # Disable built-in logging handlers (like TensorBoard)
        )

        # Use the custom logger callback
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            callbacks=[CustomLoggerCallback()],
        )

        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(self.params.output_dir)
        self.tokenizer.save_pretrained(self.params.output_dir)

    def evaluate(self):
        """Evaluate the trained model."""
        trainer = Trainer(
            model=self.model,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
        )
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        print(f"Evaluation results: {eval_results}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for sentiment analysis."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Name of the model to use.",
    )
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
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Ensure that the script runs on CPU if no GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize training parameters
    params = TrainingParams(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        eval_subset_size=args.eval_subset_size,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        device=str(device),
    )

    # Create an instance of the fine-tuner class
    fine_tuner = DistilBertFineTuner(params)

    # Load the model and tokenizer
    fine_tuner.load_model_and_tokenizer()

    # Load and tokenize the dataset
    fine_tuner.load_and_tokenize_data()

    # Train the model
    fine_tuner.train()

    # Evaluate the model
    fine_tuner.evaluate()
