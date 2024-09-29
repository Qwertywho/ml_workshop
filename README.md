# ml_workshop

## Preparation

* github repo for the workshop: [Workshop Link](https://github.com/FerdinandZhong/ml_workshop.git)
* 2 EC2 Instance (Ubuntu Based), recommended to use t2-medium
![Creation of EC2 Instance](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/ec2_instance.png?raw=true)
  *  **Instance_A**: training + inference + stress testing
  * **Instance_B**: monitoring 
  * both two instances need to be ssh accessible when launching, as all the following operations require you to execute through ssh
* huggingface account

## Training (with Instance_A)

### Environment Setting Up 

* create EC2 Instance_A
  * Ubuntu based instance
  * t2_medium (recommended)
  * inbound rules configuration (under security)
  ![Inbound Rules Configuration](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/configure_ec2_instance_inbound_rules.png?raw=true)
    * 8000 for serving the model
    * 8089 for stress testing

* ssh into the Instance_A
* install git in the instance

```bash
sudo apt-get update
sudo apt-get install -y git
``` 

* clone the git repo `git clone https://github.com/FerdinandZhong/ml_workshop.git`
* set up the environment for the training job

```shell
cd ml_workshop
bash start.sh
```

* after successfully setting up the environment, it's recommended to create a `tmux` session for the following operations.
* `tmux new -s <your session name>`

### Model Training

Use the first window of the tmux session for the training if you decide to use the tmux session.

#### install required packages

* `source ~/.bashrc`
* `fish` (optional, this cmd will activate the specific shell tool -- `fish`)
* `conda activate workshop-env`
* `pip install -e .`


#### Training

* for demo purpose, we only train a simple MLP model for the binary classification task (dataset: `imdb`)
* script for training the model: `workshop/training/training.py`
* check all the arguments needed for the training task: `python workshop/training/training.py --help`
* sample of training job: `python workshop/training/training.py --num_train_epochs 3 --batch_size 16 --input_size 10000 --hidden_size 256 `
  **Note: you can modify the parameters passed in for the training job, but the model-specific parameters should be consistant between training and inference**
* after training, we shall push trained model weights together with the tokenizer file to hub
  * a active huggingface account is required here for creating your model repo and push the files.
  * login to huggingface: `huggingface-cli login`
    * login with the token
    * token will need to be created from the huggingface website. [token creation](https://huggingface.co/settings/tokens)
    * click `Create new token` button in the page, and enable the permissions as shown in the below sample.
    ![Create new token](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_create_new_token.png?raw=true)
  * `git lfs install`
  * `python training/push_to_hub.py --repo_name <your_repo_name> --model_save_path <model_dir>`
  * the model repository should be available now in the huggingface_hub
  ![Huggingface Repo](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_repo.png?raw=true)



## Inference

### Model Serving

We will serve the trained model with FastAPI to launch a simple web server. It's recommended to execute the following serving task in the new window of the tmux session.
</br>

**create new window in the session: `ctrl+b c`**

* install the dependencies needed for the serving: 
  * `cd ml_workshop`
  * `conda activate workshop-env`
  * `pip install -e .[serve]`
* serve the app: `python workshop/inference/app.py --repo_id <your repo id> --model_input_size 10000 --model_hidden_size 256`
</br>
**Note: the input_size and hidden_size should be consistant with the training job**


### Monitoring

#### Prometheus Server

* `wget https://github.com/prometheus/prometheus/releases/download/v2.37.5/prometheus-2.37.5.linux-amd64.tar.gz`
* unzip the file

```shell
tar -xvf prometheus-2.37.5.linux-amd64.tar.gz
cd prometheus-2.37.5.linux-amd64
```

* update the `prometheus.yml`
  * `vim prometheus.yml`
  * configuration

  ```yaml
  # my global config
    global:
    scrape_interval: 15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
    evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
    # scrape_timeout is set to the global default (10s).

    # Alertmanager configuration
    alerting:
    alertmanagers:
        - static_configs:
            - targets:
            # - alertmanager:9093

    # Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
    rule_files:
    # - "first_rules.yml"
    # - "second_rules.yml"

    # A scrape configuration containing exactly one endpoint to scrape:
    # Here it's Prometheus itself.
    scrape_configs:
    # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
    - job_name: "prometheus"

        # metrics_path defaults to '/metrics'
        # scheme defaults to 'http'.

        static_configs:
        - targets: ["localhost:9090"]

    - job_name: 'fastapi-app-instance-a'
        static_configs:
        - targets: ['instance_a_public_ip:8000']
    ~                                           
  ```

* run the prometheus server: `./prometheus --config.file=prometheus.yml`

#### Grafana Dashboard

* install docker in ec2 instance
    
```shell
sudo apt update -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

* add your user to the Docker group to run Docker without `sudo`

```shell
sudo usermod -aG docker $USER
newgrp docker
```

* run the docker container for grafana dashboard
  * `docker pull grafana/grafana:latest`
  * `docker run -d -p 3000:3000 --name=grafana grafana/grafana`


### Stress Testing

####