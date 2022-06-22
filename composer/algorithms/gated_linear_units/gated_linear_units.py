# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Type, Union

import torch

try:
    from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput
    TRANSFORMERS_INSTALLED = True
except ImportError as e:
    TRANSFORMERS_INSTALLED = False

from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput, IdentityLayer
from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.models import BERTModel
from composer.utils import check_if_transformers_installed, module_surgery

log = logging.getLogger(__name__)


def from_BertOutput(layer: torch.nn.Module,
                    module_index: int,
                    act_fn: Callable,
                    gated_layer_bias: bool = False,
                    non_gated_layer_bias: bool = False) -> BERTGatedFFOutput:
    """Defines a replacement policy from a `transformers.models.bert.modeling_bert.BertOutput` to a `composer.algorithms.gated_linear_units.gated_linear_unit_layers.BERTGatedFFOutput`"""
    assert isinstance(
        layer, BertOutput
    ), 'The replacement policy requires an instance of transformers.models.bert.modeling_bert.BertOutput for the necessary fields to be defined.'
    return BERTGatedFFOutput(d_embed=layer.dense.out_features,
                             d_ff=layer.dense.in_features,
                             dropout_rate=layer.dropout.p,
                             act_fn=act_fn,
                             layernorm_eps=layer.LayerNorm.eps,
                             gated_layer_bias=gated_layer_bias,
                             non_gated_layer_bias=non_gated_layer_bias)


def from_BertIntermediate(layer: torch.nn.Module, module_index: int) -> IdentityLayer:
    """Defines a replacement policy from a `transformers.models.bert.modeling_bert.BertIntermediate` to a `composer.algorithms.gated_linear_units.gated_linear_unit_layers.IdentityLayer`"""
    return IdentityLayer()


def apply_gated_linear_units(model: torch.nn.Module,
                             optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]],
                             act_fn: Optional[Callable] = None,
                             gated_layer_bias: bool = False,
                             non_gated_layer_bias: bool = False) -> None:
    """
    Replaces the Linear layers in the feed-forward network with `Gated Linear Units <https://arxiv.org/abs/2002.05202>`_.

    Args:
        model (:class:`torch.nn.Module`): the model to modify in-place
        optimizers (:class:`torch.optim.Optimizer` | Sequence[:class:`torch.optim.Optimizer`], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
        act_fn (Callable, optional): Optionally, the activation function to use. If ``None``, the algorithm will
            use the existing activation function in the model.
        gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
        non_gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
    """
    check_if_transformers_installed(TRANSFORMERS_INSTALLED)

    # ensure that the model is an instance of a BERTModel, since our replacement policy is only defined for BERTs
    if not isinstance(model, BERTModel):
        raise ValueError('Gated Linear Units only has a surgery policy defined for instances of BERTModel.')

    if act_fn is None:
        # get the activation functions used
        act_fns = {module.intermediate_act_fn for module in model.modules() if isinstance(module, BertIntermediate)}

        if len(act_fns) != 1:
            raise ValueError('The model has non-uniform activation functions, which is currently unsupported.')

        # since our set is of length-1, let's extract the only activation function remaining.
        (act_fn,) = act_fns

    if act_fn is None:
        raise ValueError(
            'Could not find an existing activation function to use, and no custom activation function was provided.')

    # prepare the replacement policy and perform replacement
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        BertIntermediate:
            from_BertIntermediate,
        BertOutput:
            partial(from_BertOutput,
                    act_fn=act_fn,
                    gated_layer_bias=gated_layer_bias,
                    non_gated_layer_bias=non_gated_layer_bias),
    }
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of `torch.nn.LayerNorm` were found, and therefore, there were no modules to replace.'))
    log.info(f'Successfully replaced {len(replaced_instances)} of LayerNorm with a Fused LayerNorm.')


class GatedLinearUnits(Algorithm):
    """Replaces all instances of Linear layers in the feed-forward subnetwork with a `Gated Linear Unit <https://arxiv.org/abs/2002.05202>`_.
    THe Gated Linear Units provide a more expressive form for the same number of parameters, and a slight degredation to throughput.

    Runs on :attr:`~composer.core.event.Event.INIT`, so it can swap the Linear layers in the FFN for GLUs before the model is DDP wrapped.

    Args:
        act_fn (Callable, optional): Optionally, the activation function to use. If ``None``, the algorithm will
            use the existing activation function in the model.
        gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.
        non_gated_layer_bias (bool, optional): Whether to use biases in the linear layers within the GLU. Default: ``False``.

    Example:
        .. testsetup::

           model, train_dataloader, optimizer = _make_synthetic_bert_state()

        .. testcode::

           from composer.algorithms import GatedLinearUnits

           algorithm = GatedLinearUnits()
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="1ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(self,
                 act_fn: Optional[Callable] = None,
                 gated_layer_bias: bool = False,
                 non_gated_layer_bias: bool = False):
        check_if_transformers_installed(TRANSFORMERS_INSTALLED)
        self.act_fn = act_fn
        self.gated_layer_bias = gated_layer_bias
        self.non_gated_layer_bias = non_gated_layer_bias

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_gated_linear_units(model=state.model,
                                 optimizers=state.optimizers,
                                 act_fn=self.act_fn,
                                 gated_layer_bias=self.gated_layer_bias,
                                 non_gated_layer_bias=self.non_gated_layer_bias)
