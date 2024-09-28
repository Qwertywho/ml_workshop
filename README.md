# ml_workshop

## Training

### Environment Setting Up 

* create EC2 instance
* ssh into the instance
* install git by

    ```bash
    sudo apt-get update
    sudo apt-get install -y git
    ``` 

* clone the git repo
* run the start.sh

### Running Training Task

#### install required packages

* source ~/.bashrc
* conda activate workshop-env
* pip install -e .


#### Training

* python workshop/training/training.py --num_train_epochs 3 --batch_size 16 --input_size 10000 --hidden_size 256
* push model to hub
  * login to huggingface: `huggingface-cli login`
  * `git lfs install`
  * `python training/push_to_hub.py --repo_name <your_repo_name> --model_save_path <model_dir>`

#### Build Docker Image

* `sudo apt install docker.io -y`
* `sudo systemctl start docker`