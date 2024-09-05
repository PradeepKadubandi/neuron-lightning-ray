import torch
import os
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import os

#model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = LlamaForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B")
torch.save(model.state_dict(), "./llama3-8b-hf-pretrained.pt")
