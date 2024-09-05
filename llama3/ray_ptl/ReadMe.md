## LLama3 fine-tuning on single Trainium node with Ray + Pytorch Lightning

## Common Prerequisites / Setup (Needed commonly for all subsequent sections)
- Get a single trn1.32xlarge AWS Trainium EC2 instance, use the Neuron DLAMI when creating the instance (refer to ```EC2-Instance-Screenshot.png``` if you are new to this).

## Llama 3.0: Running directly from the single node Trainium instance

- Get a single trn1.32xlarge AWS Trainium EC2 instance, use the Neuron DLAMI when creating the instance (refer to ```EC2-Instance-Screenshot.png``` if you are new to this).

- Clone this repository. 
```
git clone https://github.com/PradeepKadubandi/neuron-lightning-ray.git
cd neuron-lightning-ray/llama3
```
- Setting the pyTorch env to 13.1
```
source /opt/aws_neuronx_venv_pytorch_1_13/bin/activate
```
- Install the requirements
```
pip install -r requirements.txt --extra-index-url https://pip.repos.neuron.amazonaws.com
```
- Installing Transformer version 4.32.1 (overriding the one installed automatically with requirements.txt)
```
pip install --no-warn-conflicts transformers==4.32.1
```
- Download Llama-3 8B checkpoint and save locally
```
python3 download_llama.py
```
- Create a directory for sharded checkpoints
```
mkdir -p Meta-Llama-3-8B/pretrained_weight
```
- Covert checkpoints into 8 shards to run it with TP_SIZE=8
```
python3 convert_checkpoints.py --tp_size 8 --convert_from_full_state --config config.json --input_dir llama3-8b-hf-pretrained.pt --output_dir Meta-Llama-3-8B/pretrained_weight/
```
- Run fine-tuning script
```
./tp_zero1_llama3_8b_hf_finetune_ptl.sh
```

## Llama 3.0: Using a docker container

- Build a docker container, replace ```<name>``` and ```<tag>``` to a consistent value in all the commands below.
```
cd neuron-lightning-ray/llama3 && docker build . -f Dockerfile -t llamatrnfinetune:1.0
```
- Start the docker container and login to it 
```
docker run --device=/dev/neuron0 --device=/dev/neuron1 --device=/dev/neuron2 --device=/dev/neuron3 --device=/dev/neuron4 --device=/dev/neuron5 --device=/dev/neuron6 --device=/dev/neuron7 --device=/dev/neuron8 --device=/dev/neuron9 --device=/dev/neuron10 --device=/dev/neuron11 --device=/dev/neuron12 --device=/dev/neuron13 --device=/dev/neuron14 --device=/dev/neuron15 -i -t llamatrnfinetune:1.0 /bin/bash
```
- Install transformers==4.32.1
```
pip install --no-warn-conflicts transformers==4.32.1 
```
- Download Llama-3 8B checkpoint and save locally
```
python3 download_llama.py
```
- Create a directory for sharded checkpoints
```
mkdir -p Meta-Llama-3-8B/pretrained_weight
```
- Convert checkpoints into 8 shards to run it with TP_SIZE=8
```
python3 convert_checkpoints.py --tp_size 8 --convert_from_full_state --config config.json --input_dir llama3-8b-hf-pretrained.pt --output_dir Meta-Llama-3-8B/pretrained_weight/
```
- Run fine-tuning script
```
./tp_zero1_llama3_8b_hf_finetune_ptl.sh 
```
- You can run ```neuron-top``` from another terminal by logging into the container id and seeing the memeory and utilization of accelerator cores.
