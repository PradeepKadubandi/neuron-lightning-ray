import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer, LightningDataModule
import neuronx_distributed as nxd
from neuronx_distributed.lightning import NeuronLTModule, NeuronXLAStrategy, NeuronXLAPrecisionPlugin, NeuronTQDMProgressBar, NeuronTensorBoardLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from huggingface_hub import login
import torch_xla.core.xla_model as xm
from datasets import load_dataset

def setup_neuron_environment():
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer"
    cores_to_use = "32"
    os.environ["NUM_NEURONCORES"] = cores_to_use
    os.environ["NEURON_RT_NUM_CORES"] = cores_to_use
    os.environ["TPU_NUM_DEVICES"] = cores_to_use
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = cores_to_use
    os.environ["XLA_USE_BF16"] = "1"

setup_neuron_environment()

hftoken=os.environ["HF_TOKEN"]
login(token=hftoken)

class BERTModel(NeuronLTModule):
    def __init__(self, num_labels=2):
        self.nxd_config = nxd.neuronx_distributed_config(tensor_parallel_size=32)
        self.optimizer_cls = torch.optim.AdamW
        scheduler_cls = None
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.model_fn = lambda: AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        super().__init__(self.nxd_config, self.optimizer_cls, scheduler_cls, model_fn=self.model_fn)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def training_step(self, batch, batch_idx):
        xm.mark_step()
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), lr=2e-5)

class BERTDataModule(LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        dataset = load_dataset("glue", "sst2", split="train")
        texts = dataset['sentence']
        labels = dataset['label']
        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        self.dataset = TensorDataset(encodings.input_ids, encodings.attention_mask, torch.tensor(labels))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

def main():
    model = BERTModel()
    dm = BERTDataModule(model.tokenizer)

    trainer = Trainer(
        strategy=NeuronXLAStrategy(nxd_config=model.nxd_config),
        max_epochs=3,
        plugins=[NeuronXLAPrecisionPlugin()],
        enable_checkpointing=False,
        logger=NeuronTensorBoardLogger(save_dir="logs", log_rank0=False),
        log_every_n_steps=10,
        callbacks=[NeuronTQDMProgressBar()],
    )

    trainer.fit(model=model, datamodule=dm)

    #Print the final loss
    print(f"\nFinal Training Loss: {trainer.callback_metrics.get('train_loss', 'N/A')}")

    # return results

def _mp_fn(index, args):
    main()

if __name__ == "__main__":
    _mp_fn(0, None)
