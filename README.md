# ml_workshop

## Preparation

* 2 EC2 Instance (Ubuntu Based), recommended to use t2-medium and have 29GB storage.
![Creation of EC2 Instance](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/ec2_instance.png?raw=true)
  *  **Instance_A**: training + inference + stress testing
  * **Instance_B**: monitoring 
  * both two instances need to be ssh accessible when launching, as all the following operations require you to execute through ssh
* huggingface account

## Training (with Instance_A)

### Environment Setting Up 

* Create EC2 Instance_A under your security group
  * Ubuntu based instance
  * t2_medium 
  ![t2_medium](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/t2_medium_sample.png?raw=true)
  * create the key_pair for your instances
  * set the disk space to be 29GB
  ![disk_space](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/disk_space_setting.png?raw=true)
  * append the following inbound rules configuration (under security)
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
* tmux cheetsheet for your reference (open the cheetsheet in another tab): [tmux_cheetsheet](https://tmuxcheatsheet.com/)
* `tmux new -s <anyname you prefered>`

### Model Training

Use the first window of the tmux session for the training if you decide to use the tmux session.

#### Install required packages

* `source ~/.bashrc`
* `fish` (optional, this cmd will activate the specific shell tool -- `fish`)
* `conda activate workshop-env`
* `pip install -e .`


#### Training

* for demo purpose, we only train a simple MLP (multilayer perceptron) model for the binary classification task (dataset: `imdb`)
* script for training the model (for your reference): **workshop/training/training.py**
* check all the arguments needed for the training task: `python workshop/training/training.py --help`
* sample of training job: `python workshop/training/training.py --output_dir ./results --num_train_epochs 3 --batch_size 16 --input_size 10000 --hidden_size 256`

  **Note: you can modify the parameters passed in for the training job, but the model-specific parameters should be consistant between training and inference**
* after training, we shall push trained model weights together with the tokenizer file to hub
  * create huggingface account: [huggingface main page](https://huggingface.co/)
  * an active Hugging Face account is required to create your model repository and upload files.
  * create your model repository through huggingface UI by clicking your account profile image as shown in the below figure. ![model creation](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_create_model.png?raw=true)
  * the full name (including the account name) is the id of your repository. ![model creation](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_repo_id.png?raw=true)
  * token will need to be created from the huggingface website. [token creation](https://huggingface.co/settings/tokens)
  * click `Create new token` button in the page, and enable the permissions as shown in the below sample.
  ![Create new token](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_create_new_token.png?raw=true)

    **Note: remember to save your token value locally**

  * login to huggingface in your instance terminal: `huggingface-cli login` with the token you created
  * `git lfs install`
  * `python workshop/training/push_to_hub.py --repo_name <your repo id> --model_save_path ./results`
  * the model repository should be available now in the huggingface_hub
  ![Huggingface Repo](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/huggingface_repo.png?raw=true)
  * **Do a screenshot of your uploaded files and save it for submission**



## Inference

### Model Serving

We will serve the trained model with FastAPI & Uvicorn to launch a simple web server. It's recommended to execute the following serving task in the new window of the tmux session.
</br>

**create new window in the session: `ctrl+b, c`**

(Press ctrl+b together, then release them and press c)

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
  * `cd ml_workshop` (ensure you're under this directory)
  * `conda activate workshop-env`
  * `pip install -e .[serve]`
* serve the app: `python workshop/inference/app.py --repo_id <your repo id> --model_input_size 10000 --model_hidden_size 256`</br>
**Note: the input_size and hidden_size should be consistant with the training job**

* you should be able to see the below message in your terminal![Web Server](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/fastapi_running.png?raw=true)
* you can verify your simple running application from the instance:
  * `ctrl+b, c`

  * 
    ```shell
    curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"  -d '{"text": "Your text content here"}'
    ```

* you can also verify the prometheus client is running with the endpoint `/metrics` by running the below cmd in the same tmux window
  ```shell
  curl -X GET http://localhost:8000/metrics
  ```

  you should be able to see the similar outputs as shown in the below figure
  ![Prometheus Client Raw Outputs](https://github.com/FerdinandZhong/ml_workshop/blob/main/images/prometheus_metrics_raw_outputs.png?raw=true)



### Monitoring

Now we should set up the monitoring instance in the **Instance_B**.

**Follow the same instruction for creating Instance_A**

Similarly, the inbound rules should be appended to enable access through ports

* `3000` -- port for Grafana Dashboard
* `9090` -- port for Prometheus Server


Now connect to instance_b.

It's still recommended to execute the following steps within the tmux session (use below command to create a new tmux session).

`tmux new -s monitoring`

Then running the below cmds inside the tmux window

```shell
sudo apt-get update
sudo apt-get install -y curl wget 
git clone https://github.com/FerdinandZhong/ml_workshop.git
```

#### Prometheus Server

In the first tmux window of instance_b, let's set up the prometheus server

* install the Prometheus Server as the first step:
  * `wget https://github.com/prometheus/prometheus/releases/download/v2.37.5/prometheus-2.37.5.linux-amd64.tar.gz`
  * unzip the file

    ```shell
    tar -xvf prometheus-2.37.5.linux-amd64.tar.gz
    cd prometheus-2.37.5.linux-amd64
    ```

  * update the `prometheus.yml` by
    * `cp ~/ml_workshop/prometheus.yml .`
    * run `vim prometheus.yml` or `nano prometheus.yml` to update the yaml file.
    * replace the original `instance_a_public_ip` with your **Instance A's public ip**
    
  * save the file

* run the prometheus server: `./prometheus --config.file=prometheus.yml`
* you can verify the Prometheus Serve through `http://instance_b_public_ip:9090` in your browser

#### Grafana Dashboard

Go back to your instance_b's terminal, leave the prometheus server running. Create another window in the session. `ctrl+b, c`

We wil use docker to run the Grafana Dashboard

* install docker in ec2 instance
    
```shell
sudo apt update -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

* run below cmds for add $User to docker group to facilitate running `docker command`

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

It's recommended to run the stress testing in the new window of the tmux session `ctrl+b c`.

* in the new window, we should activate the same conda environment `conda activate workshop-env`
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
