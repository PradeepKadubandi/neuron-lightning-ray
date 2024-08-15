import os
import time
from typing import Any, Dict
import ray.train
import ray.train.lightning
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.strategies import XLAStrategy
from neuronx_distributed.lightning.strategy import (
    NeuronXLAStrategy,
)
from neuronx_distributed.lightning import (
    NeuronLTModule,
    NeuronXLAPrecisionPlugin,
    NeuronTQDMProgressBar,
    NeuronTensorBoardLogger
)

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.torch.xla import TorchXLAConfig

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# XLA imports
import torch_xla.core.xla_model as xm

# XLA imports for parallel loader and multi-processing
import torch_xla.distributed.parallel_loader as pl_xla
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.plugins.environments import LightningEnvironment

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import neuronx_distributed as nxd
from pytorch_lightning import Trainer, LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

# Global constants
EPOCHS = 100
WARMUP_STEPS = 2
BATCH_SIZE = 32

class RayNeuronXLAStrategy(NeuronXLAStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # @property
    # def root_device(self) -> torch.device:
    #     print ("RayNeuronXLAStrategy TRACE: Got call for root_device")
    #     super_device = super().root_device # Raises "Accessing the XLA device before processes have spawned is not allowed."
    #     print (f"RayNeuronXLAStrategy TRACE: {super_device=}")
    #     return super_device

    # @property
    # def distributed_sampler_kwargs(self) -> Dict[str, Any]:
    #     return dict(
    #         num_replicas=self.world_size, # In TP/PP cases, this should be DP size instead of world size!
    #         rank=self.global_rank,
    #     )

    # def setup_environment(self) -> None:
    #     print ("RayNeuronXLAStrategy TRACE: Got call for setup_environment!")
    #     # super().setup_environment()
    #     self.setup_distributed()
    
    def setup_distributed(self) -> None:
        print (f"RayNeuronXLAStrategy TRACE: Got call for setup_distributed, value of {self.parallel_devices=}!")

        super(NeuronXLAStrategy, self).setup_distributed()
        # init model parallel if needed
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size=self.tensor_parallel_size,
                pipeline_model_parallel_size=self.pipeline_parallel_size,
            )

        self.data_parallel_rank = get_data_parallel_rank()
        self.data_parallel_size = get_data_parallel_size()
        self.tensor_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_parallel_rank = get_pipeline_model_parallel_rank()

class SimpleTrainiumModel(NeuronLTModule):
    def __init__(self):
        self.nxd_config = nxd.neuronx_distributed_config(tensor_parallel_size=2)
        self.optimizer_cls = torch.optim.Adam
        scheduler_cls = None
        self.model_fn = lambda: nn.Linear(10, 1)
        super().__init__(self.nxd_config, self.optimizer_cls, scheduler_cls, model_fn = self.model_fn)
        #self.layer = nn.Linear(10, 1)
        
    
    # def setup(self, stage=None):
    #     self.model = initialize_parallel_model(
    #         self.nxd_config,
    #         self.model_fn,
    #         *self.model_args,
    #         **self.model_kwargs,
    #     )
    #     self.averaged_loss = torch.zeros(1, dtype=torch.double).to(xm.xla_device())
    #     self.print_pp_rank = 0 if self.log_rank0 else self.trainer.strategy.pipeline_parallel_size - 1

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y_hat = self.layer(x)
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=0.02)

class SimpleDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        x = torch.randn(1000, 10)
        y = torch.randn(1000, 1)
        self.dataset = TensorDataset(x, y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
    
def train_func():
    # Env variables copied from https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama/tp_zero1_llama_hf_pretrain/tp_zero1_llama2_7B_hf_pretrain.sh
    # breakpoint()
    cores_to_use = "32"
    env_variables_to_set = {
        # "NEURON_CC_FLAGS": "--model-type transformer --distribution-strategy=llm-training --retry_failed_compilation",
        "NEURON_FUSE_SOFTMAX": "1",
        "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS":"3",
        # "MALLOC_ARENA_MAX":"64",
        "NUM_NEURONCORES":cores_to_use,
        "NEURON_RT_NUM_CORES":cores_to_use,
        "TPU_NUM_DEVICES":cores_to_use,
        "TPU_CHIPS_PER_HOST_BOUNDS":cores_to_use,
        # "RAY_DEBUG": "1",
    }
    for name, value in env_variables_to_set.items():
        os.environ[name] = value

    # Set up PyTorch Lightning model and data module
    model = SimpleTrainiumModel()
    dm = SimpleDataModule()

    # Set up PyTorch Lightning Trainer with TrainiumXLAStrategy
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        strategy=RayNeuronXLAStrategy(nxd_config=model.nxd_config),
        plugins=[NeuronXLAPrecisionPlugin()],
        logger=NeuronTensorBoardLogger(save_dir="logs", log_rank0=False),
        callbacks=[NeuronTQDMProgressBar()],
    )

    # Run the training loop
    print("----------Training ---------------")
    trainer.fit(model, datamodule=dm)

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_checkpoint("checkpoints/checkpoint.pt")

    print("----------End Training ---------------")


def main():
    # trn1.32xlarge -> 32 neuron_cores, 128 CPU
    # 2x trn1.32xlarge
    # os.environ["RAY_DEBUG"] = "1"
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        torch_config=TorchXLAConfig(),
        scaling_config=ScalingConfig(
            num_workers=32, resources_per_worker={"neuron_cores": 1}
        ),
    )
    result = trainer.fit()
    print(result)

if __name__ == "__main__":
    main()

