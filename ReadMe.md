## About
This repository is a contained repository to run Pytorch (PT from here on) Lightning sample on AWS Trainium. It was developed by taking only the relevant code from <a href=https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_tp_pp_ptl.html>this</a> tutorial. The original tutorial uses parallel cluster to run the workload on mulitple node cluster. This is a simpler sample that can train LLama 2.0 model on a single trn1.32xlarge instance with reduced batch size and using all cores for Tensor Parallelism.
It also contains a hacky implementation to make Ray work with PT Lightning on Trainium.

## Prerequisites / Setup
- Get a single trn1.32xlarge AWS Trainium EC2 instance, use the Neuron DLAMI when creating the instance.
- Clone this repository (the final commands to run assume the repository is closed in home directory at path /home/ubuntu. If a different path is used to clone, the commands need to be changed accordingly).
- Download the dataset. The command below downloads the dataset to the path ```~/examples_datasets/``` and the final commands to launch assume that path for dataset. If the script is modified OR the data is placed in some other path, the commands need to be adjusted as appropriate.
```
cd data && python get_dataset.py —llama-version 2
```
- Do either of the below:
  - For PT Lightning only run (torchrun launch step in ```commands.txt```), source the pytorch 2 environment in terminal.
    ```
    source /opt/aws_neuronx_venv_pytorch_2_1/bin/activate
    ```
  - If you want to run Ray with PT Lightning instead (launch using python in ```commands.txt```), source the pytorch 1 environment instead. 
    ```
    source /opt/aws_neuronx_venv_pytorch_1_13/bin/activate
    ```
- Install the requirements
```
pip install -r requirements.txt
```
- Run the relevant commands from ```commands.txt```

