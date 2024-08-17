## About
This repository is a contained repository to run Pytorch Lightning (PTL from here on) sample on AWS Trainium. It was developed by taking only the relevant code from <a href=https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tutorials/training_llama2_tp_pp_ptl.html>this</a> tutorial. The original tutorial uses parallel cluster to run the workload on mulitple node cluster. This is a simpler sample that can train LLama 2.0 7B model on a single trn1.32xlarge instance with reduced batch size and using all cores for Tensor Parallelism.
It also contains a hacky implementation to make Ray work with PTL on Trainium.

## Prerequisites / Setup (Needed for docker steps as well)
- Get a single trn1.32xlarge AWS Trainium EC2 instance, use the Neuron DLAMI when creating the instance (refer to ```EC2-Instance-Screenshot.png``` if you are new to this).
- Clone this repository (The commands below assume the repository is cloned in home directory at path /home/ubuntu. If a different path is used to clone, the commands need to be changed accordingly).
- Download the dataset. The command below downloads the dataset to the path ```/home/ubuntu/examples_datasets/``` and further commands in this ReadMe assume that path for dataset. If the script is modified OR the data is placed in some other path, the commands need to be adjusted as appropriate.
```
cd neuron-lightning-ray/data && python get_dataset.py --llama-version 2
```

## Running directly from the instance
- Do either of the below:
  - For PTL only run (torchrun launch step in ```commands.txt```), source the pytorch 2 environment in terminal.
    ```
    source /opt/aws_neuronx_venv_pytorch_2_1/bin/activate
    ```
  - If you want to run Ray with PTL instead (launch using python in ```commands.txt```), source the pytorch 1 environment instead. Note that this relies on a custom PTL strategy implementation for now. 
    ```
    source /opt/aws_neuronx_venv_pytorch_1_13/bin/activate
    ```
- Install the requirements
```
pip install -r requirements.txt
```
- Run the relevant commands from ```commands.txt```


## Using a docker container (PTL on Trainium only)
- Build a docker container, replace ```<name>``` and ```<tag>``` to a consistent value in all the commands below.
```
docker build . -f Dockerfile -t <name>:<tag>
```

- Run the command from the docker container (use same ```<name>``` and ```<tag>``` from the above) to run PTL on Trainium cores
```
docker run --device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3 --device=/dev/neuron4 --device=/dev/neuron5 --device=/dev/neuron6 --device=/dev/neuron7 --device=/dev/neuron8 --device=/dev/neuron9 --device=/dev/neuron10 --device=/dev/neuron11 --device=/dev/neuron12 --device=/dev/neuron13 --device=/dev/neuron14 --device=/dev/neuron15 -v /home/ubuntu/examples_datasets:/examples_datasets -itd <name>:<tag> torchrun --nproc_per_node 32 main.py --model_path /neuron-lightning-ray/config.json --data_dir /examples_datasets/wikicorpus_llama2_tokenized_4k --tensor_parallel_size 32 --train_batch_size 1 --max_steps 100 --warmup_steps 5 --lr 3e-4 --grad_accum_usteps 4 --seq_len 4096 --use_sequence_parallel 0 --use_selective_checkpoint 1 --use_fp32_optimizer 0 --use_zero1_optimizer 1 --scheduler_type 'linear' --use_flash_attention 0
```

- You can run ```neuron-top``` from a terminal on the host instance and see the memeory and utilization of accelerator cores.

## BERT Examples

- For PyTorch Lightning (PTL) only example run
    ```
    source /opt/aws_neuronx_venv_pytorch_2_1/bin/activate
    export HF_TOKEN="<REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN>"
    cd neuron-lightning-ray 
    torchrun --nproc_per_node=32 bert/bert-ptl.py
    ```

- For RayTrain with PTL example run
    ```
    source /opt/aws_neuronx_venv_pytorch_1_13/bin/activate
    export HF_TOKEN="<REPLACE_WITH_YOUR_HUGGINGFACE_TOKEN>"
    cd neuron-lightning-ray
    python -m bert.bert-raytrain-ptl
    ```

