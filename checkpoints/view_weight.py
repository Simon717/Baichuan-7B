import torch

model = torch.load("./Epoch-1/mp_rank_00_model_states.pt")

for k in model.keys():
    print(k)

"""
module
buffer_names
optimizer
param_shapes
frozen_param_shapes
shared_params
frozen_param_fragments
lr_scheduler
data_sampler
random_ltd
sparse_tensor_module_names
skipped_steps
global_steps
global_samples
dp_world_size
mp_world_size
ds_config
ds_version
"""

print()
for k, v in model['module'].items():
    print(f"{k}\t\tshape={v.shape}")

"""
model.embed_tokens.weight		shape=torch.Size([64000, 64])
model.layers.0.self_attn.W_pack.weight		shape=torch.Size([192, 64])
model.layers.0.self_attn.o_proj.weight		shape=torch.Size([64, 64])
model.layers.0.self_attn.rotary_emb.inv_freq		shape=torch.Size([16])
model.layers.0.mlp.gate_proj.weight		shape=torch.Size([16, 64])
model.layers.0.mlp.down_proj.weight		shape=torch.Size([64, 16])
model.layers.0.mlp.up_proj.weight		shape=torch.Size([16, 64])
model.layers.0.input_layernorm.weight		shape=torch.Size([64])
model.layers.0.post_attention_layernorm.weight		shape=torch.Size([64])
model.norm.weight		shape=torch.Size([64])
lm_head.weight		shape=torch.Size([64000, 64])
"""


## 模拟 https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main 模型
print()
torch.save(model['module'], "baichuan_hf_model.bin")
