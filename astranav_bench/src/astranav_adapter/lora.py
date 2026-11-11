# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import torch
import torch.nn as nn
GROUPS = (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')), ('mlp', ('gate_proj', 'up_proj', 'down_proj')))

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float=0.0):
        super().__init__()
        self.base, self.r, self.scale = (base, r, alpha / r)
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t() * self.scale).to(x.dtype)

def llm_layers(model):
    return model.model.language_model.layers

def apply_lora(model, r: int=32, alpha: float=64.0, dropout: float=0.05):
    n = 0
    for layer in llm_layers(model):
        for group, names in GROUPS:
            parent = getattr(layer, group)
            for nm in names:
                setattr(parent, nm, LoRALinear(getattr(parent, nm), r, alpha, dropout))
                n += 1
    return n

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def enable_vision_training(model):
    vis = model.model.visual
    for mod in (vis.conv_list, vis.merger):
        mod.float()
        for p in mod.parameters():
            p.requires_grad = True

def set_train_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    model.model.visual.dino.eval()

def trainable_params(model):
    lora = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith('.A') or n.endswith('.B'))]
    vis = [p for n, p in model.named_parameters() if p.requires_grad and '.visual.' in n]
    return (lora, vis)

def lora_state(model):
    out = {}
    for i, layer in enumerate(llm_layers(model)):
        for group, names in GROUPS:
            for nm in names:
                mod = getattr(getattr(layer, group), nm)
                out[f'{i}.{group}.{nm}'] = dict(A=mod.A.detach().float().cpu(), B=mod.B.detach().float().cpu(), scale=mod.scale)
    return out

def vision_state(model):
    vis = model.model.visual
    return dict(conv_list={k: v.detach().float().cpu() for k, v in vis.conv_list.state_dict().items()}, merger={k: v.detach().float().cpu() for k, v in vis.merger.state_dict().items()})

def save_ft(model, path, meta: dict):
    torch.save(dict(lora=lora_state(model), vision=vision_state(model), meta=meta), path)

def load_ft_into_wrapped(model, ck):
    for i, layer in enumerate(llm_layers(model)):
        for group, names in GROUPS:
            for nm in names:
                mod = getattr(getattr(layer, group), nm)
                s = ck['lora'][f'{i}.{group}.{nm}']
                mod.A.data.copy_(s['A'])
                mod.B.data.copy_(s['B'])
    vis = model.model.visual
    vis.conv_list.load_state_dict(ck['vision']['conv_list'])
    vis.merger.load_state_dict(ck['vision']['merger'])

@torch.no_grad()
def merge_ft(model, ck):
    for i, layer in enumerate(llm_layers(model)):
        for group, names in GROUPS:
            for nm in names:
                lin = getattr(getattr(layer, group), nm)
                s = ck['lora'][f'{i}.{group}.{nm}']
                delta = s['B'].to(lin.weight.device) @ s['A'].to(lin.weight.device) * s['scale']
                lin.weight.data = (lin.weight.data.float() + delta).to(lin.weight.dtype)
    vis = model.model.visual
    vis.conv_list.load_state_dict({k: v.to(next(vis.conv_list.parameters()).dtype) if v.is_floating_point() else v for k, v in ck['vision']['conv_list'].items()})
    vis.merger.load_state_dict({k: v.to(next(vis.merger.parameters()).dtype) if v.is_floating_point() else v for k, v in ck['vision']['merger'].items()})
