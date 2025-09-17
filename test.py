import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./MiniMind2", trust_remote_code=True)
torch.save(model.state_dict(), "./MiniMind2/pytorch_model.pth")
model = AutoModelForCausalLM.from_pretrained("./MiniMind2", trust_remote_code=True, use_safetensors=False)
