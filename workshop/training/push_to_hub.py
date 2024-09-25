import argparse
import os
import torch
from huggingface_hub import HfApi, Repository, create_repo

def upload_model_to_huggingface(repo_name, model_save_path):
    # Set up repository
    api = HfApi()
    repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

    # Initialize a Repository object to handle versioning and commits
    repo = Repository(local_dir=model_save_path, clone_from=repo_id)

    # Push the model to the Hugging Face Hub
    repo.push_to_hub()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument('--repo_name', type=str, required=True, help="Name of the Hugging Face repository")
    parser.add_argument('--model_save_path', type=str, required=True, help="Local path to save the model files")
    
    args = parser.parse_args()
    upload_model_to_huggingface(args.repo_name, args.model_save_path)

