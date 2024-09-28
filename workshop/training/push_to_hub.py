import argparse

from huggingface_hub import HfApi


def upload_model_to_huggingface(repo_name, model_save_path):
    # Set up repository
    api = HfApi()
    repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

    api.upload_file(
        repo_id=repo_id,
        path_in_repo="pytorch_model.bin",
        path_or_fileobj=f"{model_save_path}/pytorch_model.bin",
    )
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="config.json",
        path_or_fileobj=f"{model_save_path}/config.json",
    )
    api.upload_file(
        repo_id=repo_id,
        path_in_repo="tfidf_vectorizer.pkl",
        path_or_fileobj=f"{model_save_path}/config.json",
    )


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="Name of the Hugging Face repository",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Local path to save the model files",
    )

    args = parser.parse_args()
    upload_model_to_huggingface(args.repo_name, args.model_save_path)
