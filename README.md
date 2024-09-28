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


## Inference

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