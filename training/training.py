from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

# Load the dataset
dataset = load_dataset("imdb")

# Load the pre-trained DistilBert model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into train and test sets
train_dataset = (
    tokenized_dataset["train"].shuffle(seed=42).select(range(20000))
)  # Use subset for faster training
test_dataset = (
    tokenized_dataset["test"].shuffle(seed=42).select(range(5000))
)  # Use subset for evaluation

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# Evaluate the model
eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")
