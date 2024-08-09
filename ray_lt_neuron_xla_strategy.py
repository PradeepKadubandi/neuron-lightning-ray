import os
from typing import Dict
from neuronx_distributed.lightning.strategy import NeuronXLAStrategy
from ray.train.lightning import RayLightningEnvironment
import torch
from lightning_fabric.plugins.environments import (
    TorchElasticEnvironment,
    XLAEnvironment,
)
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.strategies import XLAStrategy
from torch import Tensor

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from neuronx_distributed.lightning.accelerator import NeuronXLAAccelerator
from neuronx_distributed.lightning.checkpoint_io import NeuronCheckpointIO
from neuronx_distributed.lightning.launcher import _NeuronXLALauncher

class RayLightningNeuronXlaStrategy(NeuronXLAStrategy):
    def __init__(
        self,
        nxd_config: Dict = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        debug: bool = False,
        sync_module_states: bool = False,
        checkpoint_io: bool = None,
        save_load_xser: bool = True,
    ):
        cluster_environment = RayLightningEnvironment()
        # The code below is copied as is from NeuronXLAStrategy and unneeded parts are commented out
        # TODO: Refactor/update the NeuronXLAStrategy to properly accept a custom environment
        # if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        #     cluster_environment = TorchElasticEnvironment()
        # else:
        #     cluster_environment = XLAEnvironment()

        super(XLAStrategy, self).__init__(
            accelerator=NeuronXLAAccelerator(),
            cluster_environment=cluster_environment,
            debug=debug,
        )

        if not checkpoint_io:
            self.checkpoint_io = NeuronCheckpointIO(save_load_xser=save_load_xser)
        elif isinstance(checkpoint_io, NeuronCheckpointIO):
            self.checkpoint_io = checkpoint_io
        else:
            raise NotImplementedError(f"NeuronXLAStrategy only supports NeuronCheckpointIO")

        self.debug = debug
        self._launched = False
        self._sync_module_states = sync_module_states

        self.nxd_config = nxd_config

        if self.nxd_config is not None:
            self.tensor_parallel_size = self.nxd_config["tensor_parallel_size"]
            self.pipeline_parallel_size = self.nxd_config["pipeline_parallel_size"]
        else:
            self.tensor_parallel_size = tensor_parallel_size
            self.pipeline_parallel_size = pipeline_parallel_size
        
        
