# ml_workshop

## Preparation

* github repo for the workshop: [Workshop Link](https://github.com/FerdinandZhong/ml_workshop.git)
* 2 EC2 Instance (Ubuntu Based), recommended to use t2-medium and have 29GB storage.
![Creation of EC2 Instance](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/ec2_instance.png?raw=true)
  *  **Instance_A**: training + inference + stress testing
  * **Instance_B**: monitoring 
  * both two instances need to be ssh accessible when launching, as all the following operations require you to execute through ssh
* huggingface account

## Training (with Instance_A)

### Environment Setting Up 

* create EC2 Instance_A
  * Ubuntu based instance
  * t2_medium 
  ![t2_medium](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/t2_medium_sample.png?raw=true)
  * create the key_pair for your instances
  * set the disk space to be 29GB
  ![disk_space](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/disk_space_setting.png?raw=true)
  * inbound rules configuration (under security)
  ![Inbound Rules Configuration](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/configure_ec2_instance_inbound_rules.png?raw=true)
    * 8000 for serving the model
    * 8089 for stress testing

* connect to your instance
![ec2_connection](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/connect_to_ec2.png?raw=true)
  * alternatively, if you're using Macbook and have your private rsa key set up before for aws, you can connect to you instance through ssh client.

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
* tmux cheetsheet for your reference: [tmux_cheetsheet](https://tmuxcheatsheet.com/)
* `tmux new -s <anyname you prefered>`

### Model Training

Use the first window of the tmux session for the training if you decide to use the tmux session.

#### install required packages

* `source ~/.bashrc`
* `fish` (optional, this cmd will activate the specific shell tool -- `fish`)
* `conda activate workshop-env`
* `pip install -e .`


#### Training

* for demo purpose, we only train a simple MLP model for the binary classification task (dataset: `imdb`)
* script for training the model: **workshop/training/training.py**
* check all the arguments needed for the training task: `python workshop/training/training.py --help`
* sample of training job: `python workshop/training/training.py --output_dir ./results --num_train_epochs 3 --batch_size 16 --input_size 10000 --hidden_size 256 `
  **Note: you can modify the parameters passed in for the training job, but the model-specific parameters should be consistant between training and inference**
* after training, we shall push trained model weights together with the tokenizer file to hub
  * a active huggingface account is required here for creating your model repo and push the files.
  * create your model repository in huggingface UI. ![model creation](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_create_model.png?raw=true)
  * the full name (including the account name) is the id of your repository. ![model creation](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_repo_id.png?raw=true)
  * login to huggingface in your instance terminal: `huggingface-cli login`
    * login with the token
    * token will need to be created from the huggingface website. [token creation](https://huggingface.co/settings/tokens)
    * click `Create new token` button in the page, and enable the permissions as shown in the below sample.
    ![Create new token](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_create_new_token.png?raw=true)
  * `git lfs install`
  * `python workshop/training/push_to_hub.py --repo_name <your repo id> --model_save_path ./results`
  * the model repository should be available now in the huggingface_hub
  ![Huggingface Repo](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_repo.png?raw=true)



## Inference

### Model Serving

We will serve the trained model with FastAPI & Uvicorn to launch a simple web server. It's recommended to execute the following serving task in the new window of the tmux session.
</br>

**create new window in the session: `ctrl+b c`**

In this practice, we simply serve the model inside the same instance. However, in the industry practice, from model training to model inference can be much more complex. 

Typically, after the model has been trained, the files will be pushed to model registry (we use huggingface hub here for simplicity).

In the other hand, we shall prepare the implementation of the model inference application and deploy it as the micro service in the target destination with the following steps:

* CI --> for testing of the implementation
* CD:
  * build docker image of the inference application
  * push to the image registry
  * deploy the docker image in the target destination
    * pulling the model weights from the model registry while launching the application.

In today's practice, due to the limitation of the resource, we don't go through the CICD process and we don't build the docker image for the inference application.
In the following steps, we manually execute the launching of the application which typically will be automated in the industry practice. For simulation purpose, when launching the application, model weights are still pulled from the huggingface hub.

There's also a sample Dockerfile for building the docker image for this simple inference application. You can try it yourself if you're interested. It's recommended to build the docker image and run the docker container in an instance with larger storage space.

* install the dependencies needed for the serving: 
  * `cd ml_workshop`
  * `conda activate workshop-env`
  * `pip install -e .[serve]`
* serve the app: `python workshop/inference/app.py --repo_id <your repo id> --model_input_size 10000 --model_hidden_size 256`</br>
**Note: the input_size and hidden_size should be consistant with the training job**
![Web Server](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/fastapi_running.png?raw=true)
* you can verify your simple running application from your local machine:

```shell
curl -X POST "http://<your public ip of instance_a>:8000/predict"   -H "Content-Type: application/json"  -d '{"text": "Your text content here"}'
```

* you can also verify the prometheus client is running with the endpoint `/metrics` by `curl -X GET http://<your public ip of instance_a>:8000/metrics`
![Prometheus Client Raw Outputs](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/prometheus_metrics_raw_outputs.png?raw=true)


### Monitoring

Now we should set up the monitoring instance in the **Instance_B**.

Similarly, the inbound rules should be set up to enable access through ports

* `3000` -- port for Grafana Dashboard
* `9090` -- port for Prometheus Server

It's still recommended to execute the following steps within the tmux session.

```shell
sudo apt-get update
sudo apt-get install -y tmux curl wget 
tmux new -s monitoring
```

#### Prometheus Server

In the first window, let's set up the prometheus server

* install the Prometheus Server as the first step:
  * `wget https://github.com/prometheus/prometheus/releases/download/v2.37.5/prometheus-2.37.5.linux-amd64.tar.gz`
  * unzip the file

    ```shell
    tar -xvf prometheus-2.37.5.linux-amd64.tar.gz
    cd prometheus-2.37.5.linux-amd64
    ```

  * update the `prometheus.yml`
  * `vim prometheus.yml`
  * update the configuration file as below, type `i` to edit the file

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
  * save the file by `esc + :x`

* run the prometheus server: `./prometheus --config.file=prometheus.yml`
* you can verify the Prometheus Serve through `http://instance_b_public_ip:9090` in your browser

#### Grafana Dashboard

Create another window in the session.

We wil use docker to run the Grafana Dashboard

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

* go to the dashboard UI by entering `http://instance_b_public_ip:3000` in your browser.
* username and password are both `admin`
* add the data source as your Prometheus Server
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/dashboard_add_data_source.png?raw=true)
* select Prometheus as the data source
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/select_prometheus_as_data_source.png?raw=true)
* pointing the instance_b's Prometheus Server (use your own instance_b's public ip)
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/pointing_to_prometheus_server.png?raw=true)
* create the dashboard with the data source as the Prometheus
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/create_dashboard.png?raw=true)
* create panels with the PromQL that querying over the metrics we set up in the application, below are two samples.
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/adding_histogram_query.png?raw=true)
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/adding_qps_query.png?raw=true)
* check the dashboard
![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/grafana_dashboard.png?raw=true)



### Stress Testing

Now let's go back to the **Instance_A** to start the stress testing over our launched application.

It's recommended to run the stress testing in the new window of the tmux session.

* in the new window, we should activate the same conda environment.
* dependencies required for stress testing have been installed in the last step, the package we need for the stress testing is `locust`, you can verify the installation by `pip freeze | grep locust`
* in the same instance, the application is still running under the port `8000`, therefore we are going to stress testing against the endpoint `/predict` through this port
* `locust -f workshop/stress_testing/locustfile.py --host http://0.0.0.0:8000`
* now we can start the stress testing through the UI of `locust` through the URL `http://instance_a_public_ip:8089`
  * trigger the stress testing
  ![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/start_locust_test.png?raw=true)
  * view the stress testing charts from locust UI
  ![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/stress_testing_charts.png?raw=true)
  * we can check the Grafana dashboard to monitoring the metrics of the application as well
  ![Adding Data Source](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/grafana_dashboard.png?raw=true)
