# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
import transformers
import yahp as hp
from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn.functional import relu
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
from xformers.triton.layer_norm import FusedLayerNorm as TritonLayerNorm

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

from .norms import RMSNorm

log = logging.getLogger(__name__)


@dataclass
class ActFnSearchHparams(AlgorithmHparams):
    """See :class:`Primer`"""
    act_fn_name: str = hp.required("The name of the activation function to use.")
    use_gated: bool = hp.required("Whether to use a GLU unit or a regular unit.")
    use_rmsnorm: bool = hp.required("Whether to use RMSNorm instead of LayerNorm.")
    use_fln: bool = hp.required("Whether to use fused layernorms.")
    use_triton: bool = hp.required("Whether to use fused layernorms.")

    def initialize_object(self) -> "Primer":
        return ActFnSearch(**asdict(self))


@torch.jit.script
def squared_relu(x):
    return F.relu(x)**2


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
                           ).read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.96)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


def apply_act_fn(model: torch.nn.Module, optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                 act_fn_name: str, use_gated: bool, use_rmsnorm: bool, use_fln: bool, use_triton: bool) -> None:
    act_fns = {
        "squared_relu": squared_relu,
        "fast_gelu": transformers.activations.gelu_fast,
        "gelu": torch.nn.functional.gelu,
        "relu": torch.nn.ReLU(),
        "swish": transformers.activations.silu,
        "no_replacement": None,
    }
    act_fn = act_fns[act_fn_name]

    # get new parameter values
    d_ffs = []
    d_embeds = []
    layernorm_eps = []
    dropout_rates = []
    act_fns = []
    for idx in range(len(model.module.bert.encoder.layer)):
        bert_layer = model.module.bert.encoder.layer[idx]
        d_ffs.append(bert_layer.intermediate.dense.out_features)
        d_embeds.append(bert_layer.intermediate.dense.in_features)
        layernorm_eps.append(bert_layer.output.LayerNorm.eps)
        dropout_rates.append(bert_layer.output.dropout.p)
        act_fns.append(bert_layer.intermediate.intermediate_act_fn)

    for l in [d_ffs, d_embeds, layernorm_eps, dropout_rates, act_fns]:
        assert len(set(l)) == 1

    d_ff = d_ffs[0]
    d_ff = round((2.0 / 3.0) * d_ff)  # scale down d_ff by 1/3 in order to maintain equal number of parameters
    d_embed = d_embeds[0]
    layernorm_eps = layernorm_eps[0]
    dropout_rate = dropout_rates[0]
    if act_fn is None:
        act_fn = act_fns[0]

    if act_fn is not None and not use_gated:
        for idx in range(len(model.module.bert.encoder.layer)):
            model.module.bert.encoder.layer[idx].intermediate.intermediate_act_fn = act_fn

    if use_gated:
        policy = {
            BertIntermediate:
                lambda x, module_index: DummyBERTIntermediateOutput(),
            BertOutput:
                lambda x, module_index: BERTGatedOutput(
                    d_embed=d_embed, d_ff=d_ff, dropout_rate=dropout_rate, act_fn=act_fn, layernorm_eps=layernorm_eps)
        }
        module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)

    if use_rmsnorm:
        policy = {torch.nn.LayerNorm: lambda x, module_index: RMSNorm(dim=d_embed, eps=layernorm_eps)}
        module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)

    if use_fln and use_triton:
        raise ValueError("Cannot use both FLN and OneFlow!")

    if use_fln:
        policy = {
            torch.nn.LayerNorm: lambda x, module_index: FusedLayerNorm(normalized_shape=d_embed, eps=layernorm_eps)
        }
        module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)

    if use_triton:
        policy = {
            torch.nn.LayerNorm: lambda x, module_index: TritonLayerNorm(normalized_shape=d_embed, eps=layernorm_eps)
        }
        module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)

    print(model)


class SquaredReLU(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)**2

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class DummyBERTIntermediateOutput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return hidden_states


class BERTGatedOutput(torch.nn.Module):

    def __init__(self, d_embed, d_ff, dropout_rate, act_fn, layernorm_eps):
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_embed, d_ff)
        self.wi_1 = torch.nn.Linear(d_embed, d_ff)
        self.wo = torch.nn.Linear(d_ff, d_embed)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = act_fn
        self.layernorm = torch.nn.LayerNorm(d_embed, eps=layernorm_eps)

    def forward(self, hidden_states, input_tensor):
        # compute the activation
        hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class ActFnSearch(Algorithm):

    def __init__(self, act_fn_name: str, use_gated: bool, use_rmsnorm: bool, use_fln: bool, use_triton: bool) -> None:
        self.act_fn_name = act_fn_name
        self.use_gated = use_gated
        self.use_rmsnorm = use_rmsnorm
        self.use_fln = use_fln
        self.use_triton = use_triton

    def match(self, event: Event, state: State) -> bool:
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Replace model's existing attention mechanism with AliBi
        """

        if event == Event.INIT:
            assert state.model is not None
            apply_act_fn(state.model,
                         optimizers=state.optimizers,
                         act_fn_name=self.act_fn_name,
                         use_gated=self.use_gated,
                         use_rmsnorm=self.use_rmsnorm,
                         use_fln=self.use_fln,
                         use_triton=self.use_triton)
