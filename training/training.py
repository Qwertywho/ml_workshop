from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
import torch


class DistilBertFineTuner:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer."""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

    def load_and_tokenize_data(self, dataset_name="imdb", max_length=512, subset_size=20000):
        """Load dataset and tokenize the text."""
        dataset = load_dataset(dataset_name)
        
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=max_length
            )

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        self.train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(subset_size))
        self.test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(5000))

    def train(self, output_dir="./results", num_train_epochs=2, batch_size=8):
        """Set training arguments and train the model."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def evaluate(self):
        """Evaluate the trained model."""
        trainer = Trainer(
            model=self.model,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
        )
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    # Ensure that the script runs on CPU if no GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the fine-tuner class
    fine_tuner = DistilBertFineTuner()

    # Load the model and tokenizer
    fine_tuner.load_model_and_tokenizer()

    # Load and tokenize the dataset
    fine_tuner.load_and_tokenize_data()

    # Train the model
    fine_tuner.train()

    # Evaluate the model
    fine_tuner.evaluate()

