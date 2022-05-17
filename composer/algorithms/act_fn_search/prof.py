import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

import time

import oneflow as flow
from apex.normalization import FusedLayerNorm
from oneflow.nn.modules.normalization import LayerNorm as OneFlowLayerNorm
from xformers.triton.layer_norm import FusedLayerNorm as TritonLayerNorm

# Create data
x = torch.randn(4000, 128, 768, device='cuda')

# upstream layernorm
norm = nn.LayerNorm(x.size()[-1]).cuda()

# cudnn warmup
for _ in range(50):
    _ = norm(x)

nb_iters = 1000
torch.cuda.synchronize()
t0 = time.time()

for _ in range(nb_iters):
    _ = norm(x)

torch.cuda.synchronize()
t1 = time.time()

print('upstream layernorm {:.3f}'.format(t1 - t0))

# apex fusedlayernorm
fused_norm = FusedLayerNorm(x.size()[-1]).cuda()

# cudnn warmup
for _ in range(50):
    _ = fused_norm(x)

nb_iters = 1000
torch.cuda.synchronize()
t0 = time.time()

for _ in range(nb_iters):
    _ = fused_norm(x)

torch.cuda.synchronize()
t1 = time.time()

print('apex layernorm {:.3f}'.format(t1 - t0))

# triton fusedlayernorm
triton_norm = TritonLayerNorm(x.size()[-1]).cuda()

# cudnn warmup
for _ in range(50):
    _ = triton_norm(x)

nb_iters = 1000
torch.cuda.synchronize()
t0 = time.time()

for _ in range(nb_iters):
    _ = triton_norm(x)

torch.cuda.synchronize()
t1 = time.time()

print('triton layernorm {:.3f}'.format(t1 - t0))

# oneflow
oneflow_norm = OneFlowLayerNorm(x.size()[-1]).cuda()
x = flow.randn(4000, 128, 768, device='cuda')

# cudnn warmup
for _ in range(50):
    _ = oneflow_norm(x)

nb_iters = 1000
torch.cuda.synchronize()
t0 = time.time()

for _ in range(nb_iters):
    _ = oneflow_norm(x)

torch.cuda.synchronize()
t1 = time.time()

print('oneflow layernorm {:.3f}'.format(t1 - t0))
