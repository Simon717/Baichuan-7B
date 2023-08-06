import os

import torch

num_layers = 1  # 提取前x层

hf_model = torch.load(r"D:\trans\code2023\LLM\Baichuan-7B\checkpoints\Epoch-1\mp_rank_00_model_states.pt")
ds_model = dict()

ds_module = dict()
for k, v in hf_model['module'].items():
    if 'layers' not in k:
        ds_module[k] = v
        continue

    if int(k.split('.')[2]) < num_layers:
        ds_module[k] = v

ds_model['module'] = ds_module
ds_model['dp_world_size'] = 0
print(ds_module.keys())

model_dir = f"hf2as_layer{num_layers}"
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "latest"), 'w') as f:
    f.write("release")
model_dir = os.path.join(model_dir, "release")
os.makedirs(model_dir, exist_ok=True)
torch.save(ds_model, os.path.join(model_dir, "mp_rank_00_model_states.pt"))
