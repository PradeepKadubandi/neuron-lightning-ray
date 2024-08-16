FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.19.1-ubuntu20.04

WORKDIR /
RUN git clone https://github.com/PradeepKadubandi/neuron-lightning-ray.git

WORKDIR /neuron-lightning-ray
RUN pip install -r requirements.txt